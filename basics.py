import torch
import math
import logging
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import networks
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




