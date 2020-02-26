import torch.nn as nn
import torch.nn.functional as F
import logging
import itertools
from . import utils
import math
import torch
#from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel


class Transformer1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(Transformer1, self).__init__()
        self.output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        nhead=1
        hidden_dim = embedding_dim
        n_layers = 1
        dropout=0.1
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = state.learnable_embedding
        self.decoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward=hidden_dim, dropout=dropout, activation='relu')
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)
        self.classifier_head = nn.Linear(hidden_dim, self.output_dim)
        #self.sigm=nn.Sigmoid()
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        tgt_size=[i for i in out.size()]
        tgt_size[-2]=1
        #print(tgt_size)
        tgt=torch.rand(tgt_size)
        hidden = self.decoder(tgt, out).squeeze(1)
        return self.classifier_head(hidden)
 