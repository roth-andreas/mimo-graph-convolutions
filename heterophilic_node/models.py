import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, dropout_edge
#from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_scatter import scatter

import LMGC, FAGCN, ACM
import torch_geometric as pyg


class SimpleModel(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_layers, conv_type, heads, dropout):
        super().__init__()
        self.enc = nn.Linear(in_dim, h_dim)
        self.convs = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            if conv_type == 'lmgc':
                conv = LMGC.LMGC(h_dim, h_dim, heads=out_dim, add_self_loops=False)
            elif conv_type == 'fagcn':
                conv = FAGCN.FAGCN(h_dim, h_dim, add_self_loops=True)
            elif conv_type == 'gatv2':
                conv = pyg.nn.GATv2Conv(h_dim, h_dim, heads=heads, concat=False)
            elif conv_type == 'gin':
                conv = pyg.nn.GINConv(nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim)))
            elif conv_type == 'acm':
                conv = ACM.ACM(h_dim, h_dim)
            else:
                print('Conv type not found!')
            self.convs.append(conv)

        self.dec = nn.Linear(h_dim, out_dim)
        self.num_layers = num_layers

    def forward(self, data):
        x = self.enc(data.x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = x + self.convs[i](x, data.edge_index)
            x = torch.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.dec(x)

