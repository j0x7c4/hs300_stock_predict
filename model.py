# encoding:utf-8
from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functions import (
    move_to_gpu
)


class TransformerModel(nn.Module):

    def __init__(self, dim_in, units,  nhead, dim_out, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.emb = nn.Linear(dim_in, units)
        encoder_layers = TransformerEncoderLayer(units, nhead, dim_out, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.optimizer = optim.Adam(
                chain(self.transformer_encoder.parameters(),
                      self.emb.parameters()),
                lr=0.01, betas=(0.5, 0.999))
        self.cuda = False if not torch.cuda.is_available() else True
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.emb(src)
        output = self.transformer_encoder(self.activation(src))
        return output

    def train(self, batch):
        Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        label = batch["label"]
        label = move_to_gpu(label)
        src = Variable(batch['x'].type(Tensor))
        logits = self.forward(src)
        loss_C = nn.CrossEntropyLoss()(logits, label)
        loss += loss_C
        loss.backward(retain_graph=True)
        self.optimizer.step()
        metrics = {
            "loss_C": loss_C.item(),
        }
        return metrics
