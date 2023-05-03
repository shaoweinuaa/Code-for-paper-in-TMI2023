from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn.pool.topk_pool import topk#,filter_adj
import torch
from torch import Tensor

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))

def filter_adj(edge_index, edge_attr, perm, num_nodes):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i  
    mask1 = mask.clone().detach()
    row, col = edge_index
    row, col = mask[row], mask[col] 
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr,mask1


class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GATConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)
        self.non_linearity = non_linearity
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr,mask = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, mask
