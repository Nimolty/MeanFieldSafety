# -*- coding: utf-8 -*-
"""
Created on Sat May 14 22:58:55 2022

@author: lenovo
"""
import torch
import torch.nn as nn 
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class EdgeConv_norm(MessagePassing):
    def __init__(self, p):
        super().__init__(aggr='mean')
        self.p = p
    
    def forward(self, x, edge_index):
        # x has shape[N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        return -(x_j - x_i) / (torch.sum((x_j - x_i)**2, axis = 1, keepdims=True).pow(self.p/2))

class GaussEdgeConv(MessagePassing):
    def __init__(self, h, is_norm):
        super().__init__(aggr='mean')
        self.h = h
        self.is_norm = is_norm
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        exp = torch.exp(-torch.abs(x_j - x_i)/self.h)
        if self.is_norm:
            return - torch.sign(x_j - x_i) * exp
        else:
            return - (x_j - x_i) * exp


class P_update(nn.Module):
    def __init__(self, p, h=1, is_norm=True):
        super().__init__()
        self.p = p
        self.h = h
        self.is_norm = is_norm
        self.conv_spatial = EdgeConv_norm(self.p)
        #self.conv_spatial = GaussEdgeConv(self.h, self.is_norm)
    
    def forward(self, state_inp, ratio):
        x, edge_index, batch = state_inp.x, state_inp.edge_index, state_inp.batch
        out = ratio * self.conv_spatial(x, edge_index)
        return out


























