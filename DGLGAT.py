import numpy as np
import torch
import dgl
import torch.nn as nn
from dgl.nn.pytorch import GATConv
from sagpool import SAGPool
from dgl.nn.pytorch import AvgPooling as gap, MaxPooling as gmp , GlobalAttentionPooling
import copy
import torch.nn.functional as F

class GatNet1(nn.Module):
    def __init__(self, input, hiddens, classifier):
        super().__init__()      
        self.gats = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(hiddens)):
            if i == 0:
                
                self.gats.append(GATConv(
                    in_feats=input, out_feats=hiddens[i][0], num_heads=hiddens[i][1], negative_slope=0.4, allow_zero_in_degree=True))
                self.pools.append(
                    SAGPool(hiddens[i][0]*hiddens[i][1], ratio=hiddens[i][2]))
            else:
                self.gats.append(GATConv(in_feats=hiddens[i-1][0]*hiddens[i-1][1], out_feats=hiddens[i]
                                 [0], num_heads=hiddens[i][1], negative_slope=0.4, allow_zero_in_degree=True))
                self.pools.append(
                    SAGPool(hiddens[i][0]*hiddens[i][1], ratio=hiddens[i][2]))

        self.globalPool = GlobalAttentionPooling(nn.Linear(hiddens[-1][0], 1))
        self.classifier = classifier

    
    def forward(self, g):
        edges = []
        res = g.ndata['feature']
        for i in range(len(self.gats)-1):
            res = self.gats[i](g, res).flatten(1)
            g, res, _ = self.pools[i](g, res)
            pass
        
        res, atten = self.gats[-1](g, res, get_attention=True)
        edges.append(g.edata['strength'])
        res = res.mean(1)
       
        res = self.globalPool(g, res)
        lbl_pred = self.classifier(res)
        return g, res, lbl_pred, atten, edges
