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
        # 接受hidden参数：
        # hidden = [ GATConv的输出维度，GATConv的多投数，SAGPool的ratio(把结点数变为 结点数*ratio 个) ]
        super().__init__()
        # 下一层的in_feats=上一层的out_feats * 上一层的num_heads
        self.gats = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(hiddens)):
            if i == 0:
                # 激活函数
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

    # 新的特征矩阵arr[m*b] = GATConv(输入特征维度a，输出特征维度b，多投数c)(图g，特征矩阵[m*n])
    # 新的图g2, 新特征矩阵2 = SAGPool(输入结点数m,保留结点的比例k)(图g,arr[m*b])
    def forward(self, g):
        edges = []
        res = g.ndata['feature']
        for i in range(len(self.gats)-1):
            res = self.gats[i](g, res).flatten(1)
            g, res, _ = self.pools[i](g, res)
            pass

        # atten 获取注意力系数
        # 最后一层gat
        res, atten = self.gats[-1](g, res, get_attention=True)
        edges.append(g.edata['strength'])
        res = res.mean(1)

        # 全局池化把k个结点变为1个结点
        res = self.globalPool(g, res)
        lbl_pred = self.classifier(res)
        return g, res, lbl_pred, atten, edges
