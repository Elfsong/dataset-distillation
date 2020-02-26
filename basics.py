import torch
import math
import logging
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from base_options import options
import networks
from itertools import chain
from collections import OrderedDict, namedtuple
from utils import pm, nan
from typing import Optional


# train models
def load_train_models(state):
    if state.train_nets_type == 'known_init':
        return networks.get_networks(state, N=state.local_n_nets)
    else:
        raise ValueError("train_nets_type: {}".format(state.train_nets_type))

def task_loss(state, output, label, **kwargs):
    if state.num_classes == 2:
        label = label.to(output, non_blocking=True).view_as(output)
        return F.binary_cross_entropy_with_logits(output, label, **kwargs) 
    else:
        return F.cross_entropy(output, label.long().argmax(-1), **kwargs)


def task_loss_eval(state, output, label, **kwargs):
    if state.num_classes == 2:
        label = label.to(output, non_blocking=True).view_as(output)
        return F.binary_cross_entropy_with_logits(output, label, **kwargs)
    else:
        return F.cross_entropy(output, label, **kwargs)


def final_objective_loss(state, output, label):
    if state.mode in {'distill_basic', 'distill_adapt'}:
        return task_loss_eval(state, output, label)
    else:
        raise NotImplementedError('mode ({}) is not implemented'.format(mode))


# NB: This trains params or model inplace!!!
def train_steps_inplace(state, models, steps, params=None, callback=None):
    if isinstance(models, torch.nn.Module):
        models = [models]
    if params is None:
        params = [m.get_param() for m in models]

    for i, (data, label, lr) in enumerate(steps):
        data = data.detach()
        label = label.detach()
        lr = lr.detach()

        for model, w in zip(models, params):
            model.train()  # callback may change model.training so we set here
            model.distilling_flag=True
            output = model.forward_with_param(data, w)
            #print(output[0].size())
            loss = task_loss(state, output, label)
            lr=lr.squeeze()
            #print(lr)
            #print(lr.size())
            loss.backward(lr)
            with torch.no_grad():
                w.sub_(w.grad)
                w.grad = None



    return params




def fixed_width_fmt(num, width=4, align='>'):
    if math.isnan(num):
        return '{{:{}{}}}'.format(align, width).format(str(num))
    return '{{:{}0.{}f}}'.format(align, width).format(num)[:width]


def _desc_step(state, steps, i):
    if i == 0:
        return 'before steps'
    else:
        lr = steps[i - 1][-1]
        return 'step {:2d} (lr={})'.format(i, fixed_width_fmt(lr.sum().item(), 6))




def infinite_iterator(iterable):
    while True:
        yield from iter(iterable)


# See NOTE [ Evaluation Result Format ] for output format

