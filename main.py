from __future__ import print_function
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import heapq
import functools
import networks
import logging
from base_options import options
import utils
import numpy as np
from networks.utils import print_network
from utils.baselines import encode
from utils.io import vis_results, load_results, save_test_results
from collections import defaultdict
from contextlib import contextmanager
import train_distilled_image




def main(state):
        logging.info('mode: {}, phase: {}'.format(state.mode, state.phase))

        # train models
        def load_train_models():
                return networks.get_networks(state, N=state.local_n_nets)

        # only construct when in training mode or test_nets_type == same_as_train
        state.models = load_train_models()

        state.test_models = state.models
        

        logging.info('Train {} steps iterated for {} epochs'.format(state.distill_steps, state.distill_epochs))
        steps = train_distilled_image.distill(state, state.models)
        print ("END \n\n\n\n\n")
            



if __name__ == '__main__':
    try:
        main(options.get_state())
    except Exception:
        logging.exception("Fatal error:")
        raise
