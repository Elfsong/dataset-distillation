from __future__ import print_function
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import heapq
import functools
import networks
from base_options import options
import numpy as np
from contextlib import contextmanager
import train_distilled_image




def main(state):

        # train models
        def load_train_models():
                return networks.get_networks(state, N=state.local_n_nets)

        # only construct when in training mode or test_nets_type == same_as_train
        state.models = load_train_models()

        state.test_models = state.models
        

        steps = train_distilled_image.distill(state, state.models)
            



if __name__ == '__main__':
    try:
        main(options.get_state())
    except Exception:
        raise
