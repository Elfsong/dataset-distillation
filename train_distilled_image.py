import time
import os
import logging
import random
import math
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import networks
from itertools import repeat, chain
from networks.utils import clone_tuple
from utils.distributed import broadcast_coalesced, all_reduce_coalesced
from utils.io import save_results
from utils.label_inits import distillation_label_initialiser
from basics import task_loss, final_objective_loss, evaluate_steps
from contextlib import contextmanager
import psutil

import faulthandler
faulthandler.enable()

class Trainer(object):
    def __init__(self, state, models):
        self.state = state
        self.models = models
        self.num_data_steps = state.distill_steps  # how much data we have
        self.T = state.distill_steps * state.distill_epochs  # how many sc steps we run
        self.num_per_step = state.num_distill_classes * state.distilled_images_per_class_per_step
        assert state.distill_lr >= 0, 'distill_lr must >= 0'
        assert len(state.init_labels)==state.num_distill_classes, 'len(init_labels) must == num_distill_classes'
        self.init_data_optim()

    def init_data_optim(self):
        self.params = []
        state = self.state
        optim_lr = state.lr
        req_lbl_grad = False
        # labels
        self.labels = []
        

        if state.num_classes==2:
                    dl_array = [[i==j for i in range(1)]for j in state.init_labels]*state.distilled_images_per_class_per_step
        else:
                    dl_array = [[i==j for i in range(state.num_classes)]for j in state.init_labels]*state.distilled_images_per_class_per_step
        distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=state.device)
        self.labels.append(distill_label)

        # data
        self.data = []

        distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.ninp,
                                   device=state.device, requires_grad=(not state.freeze_data))
        self.data.append(distill_data)


        # lr

        # undo the softplus + threshold
        raw_init_distill_lr = torch.tensor(state.distill_lr, device=state.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr
        self.params.append(self.raw_distill_lrs)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed

        for p in self.params:
            p.grad = torch.zeros_like(p)

    def get_steps(self):
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in zip(self.data, self.labels))
        lrs = self.raw_distill_lrs.unbind()
        steps = []
        for (data, label), lr in zip(data_label_iterable, lrs):
            steps.append((data,label, lr))
        return steps

    def forward(self, model, rdata, rlabel, steps):
        state = self.state

        # forward
        model.train()
        w = model.get_param()
        params = [w]
        gws = []
        for step_i, (data, label, lr) in enumerate(steps):
            with torch.enable_grad():
                model.distilling_flag=True
                output = model.forward_with_param(data, w)
                loss = task_loss(state, output, label)
                lr = lr.squeeze()
                #print(lr.size())
            gw, = torch.autograd.grad(loss, w, lr, create_graph=True)

            with torch.no_grad():
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # final L
        model.eval()
        model.distilling_flag=False
        output = model.forward_with_param(rdata, params[-1])
        ll = final_objective_loss(state, output, rlabel)
        return ll, (ll, params, gws)

    def backward(self, model, rdata, rlabel, steps, saved_for_backward):
        l, params, gws = saved_for_backward
        state = self.state

        datas = []
        gdatas = []
        lrs = []
        glrs = []
        labels=[]
        glabels=[]
        dw, = torch.autograd.grad(l, (params[-1],), create_graph=True)

        # backward
        model.train()
        # Notation:
        #   math:    \grad is \nabla
        #   symbol:  d* means the gradient of final L w.r.t. *
        #            dw is \d L / \dw
        #            dgw is \d L / \d (\grad_w_t L_t )
        # We fold lr as part of the input to the step-wise loss
        #
        #   gw_t     = \grad_w_t L_t       (1)
        #   w_{t+1}  = w_t - gw_t          (2)
        #
        # Invariants at beginning of each iteration:
        #   ws are BEFORE applying gradient descent in this step
        #   Gradients dw is w.r.t. the updated ws AFTER this step
        #      dw = \d L / d w_{t+1}
        for (data, label, lr), w, gw in reversed(list(zip(steps, params, gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            hvp_in = [w]

            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,),
                retain_graph=False
            )
            

        return datas, gdatas, lrs, glrs, labels, glabels

    def save_results(self, steps=None, visualize=True, subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(self.state, steps, visualize=visualize, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        state = self.state
        device = state.device
        niter = 10
        for epoch in range(state.epochs):
            print("Training Epoch: {}".format(epoch))
            prefetch_it = max(0, niter - 2)
            for it in range(niter):
                    data = torch.randint(5000, (1,state.maxlen))
                    target = torch.randint(2, (1,1))
                    val=(data,target)
                    yield epoch, it, val

    def train(self):
        state = self.state
        device = state.device

        
        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            
            #if it == 0 or epoch == 0:
            #    with torch.no_grad():
            #        steps = self.get_steps()
                #evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))
                    #If this block is commented out then need to set line 174 retain_graph=False
            

            tmodels = self.models
            

            losses = []
            steps = self.get_steps()

            # activate everything needed to run on this process
            grad_infos = []
            for model in tmodels:

                l, saved = self.forward(model, rdata, rlabel, steps)
                #losses.append(l)
                
                next_ones = self.backward(model, rdata, rlabel, steps, saved)
                #grad_infos.append(next_ones)
            
            #self.accumulate_grad(grad_infos)

            # all reduce if needed
            # average grad
            
            


        with torch.no_grad():
            steps = self.get_steps()
        return steps


def distill(state, models):
    return Trainer(state, models).train()
