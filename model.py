# encoding:utf-8
from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from losses import EMDLoss 
from functions import (
    move_to_gpu,
    move_to_cpu
)



class TransformerModel(nn.Module):

    def __init__(self, dim_in, units,  nhead, dim_out, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.emb = nn.Linear(dim_in, units)
        encoder_layers = TransformerEncoderLayer(units, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(units, dim_out)
        self.optimizer = optim.Adam(
                chain(self.transformer_encoder.parameters(),
                      self.emb.parameters(),
                      self.decoder.parameters()),
                lr=0.01, betas=(0.5, 0.999))
        self.cuda = False if not torch.cuda.is_available() else True
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor
        self.softmax = nn.LogSoftmax(dim=1)
        # self.loss_fn = EMDLoss()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, src):
        src = self.emb(src)
        encoder_output = self.transformer_encoder(self.activation(src))
        output = self.decoder(encoder_output.mean(dim=1))
        # output = self.decoder(encoder_output[:,0])
        return output

    def train(self, batch):
        label = batch["label"]
        if self.cuda:
            target = move_to_gpu(label)
        else:
            target = label
        src = Variable(batch['x'].type(self.Tensor))
        logits = self.forward(src)
        loss = self.loss_fn(logits, target)
        self.optimizer.zero_grad()
        # logits.retain_grad()
        loss.backward()
        # print("grad", logits.grad)
        self.optimizer.step()
        metrics = {
            "loss": loss.item(),
        }
        return metrics

    def predict(self, batch):
        src = Variable(batch['x'].type(self.Tensor))
        logits = self.forward(src)
        scores = self.softmax(logits)
        scores = torch.exp(scores)
        return scores
