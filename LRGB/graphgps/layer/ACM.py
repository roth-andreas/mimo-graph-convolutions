import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as pyg
class ACM(Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super(ACM, self).__init__()
        (
            self.in_features,
            self.out_features,
        ) = (
            in_features,
            out_features,
        )
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0

        self.gcn_low = pyg.nn.GCNConv(
            in_features,
            out_features,
            add_self_loops=False,
            normalize=False
        )
        self.gcn_high = pyg.nn.GCNConv(
            in_features,
            out_features,
            add_self_loops=False,
            normalize=False
        )
        self.gcn_mlp = pyg.nn.Linear(
            in_features,
            out_features,
        )

        self.weight_low, self.weight_high = (
            Parameter(torch.FloatTensor(in_features, out_features)),
            Parameter(torch.FloatTensor(in_features, out_features))
        )
        self.att_vec_low, self.att_vec_high = (
            Parameter(torch.FloatTensor(1 * out_features, 1)),
            Parameter(torch.FloatTensor(1 * out_features, 1))
        )
        self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features))
        self.att_vec_mlp = Parameter(torch.FloatTensor(1 * out_features, 1))

        self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.weight_low.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_low.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention2(self, output_low, output_high):
        T = 2
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None]


    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, x, edge_index):
        _, edge_weight_low = pyg.nn.conv.gcn_conv.gcn_norm(edge_index, edge_weight=None, num_nodes=x.size(0), add_self_loops=False)
        edge_weight_high = 1 - edge_weight_low
        edge_index_high, edge_weight_high = pyg.utils.add_self_loops(edge_index, edge_weight_high, num_nodes=x.size(0))
        output_low = F.relu(self.gcn_low(x, edge_index, edge_weight_low))
        output_high = F.relu(self.gcn_high(x, edge_index_high, edge_weight_high))
        output_mlp = F.relu(self.gcn_mlp(x))

        #self.att_low, self.att_high = self.attention2(
        #    (output_low), (output_high)
        #)
        
        self.att_low, self.att_high, self.att_mlp = self.attention3(
            (output_low), (output_high), (output_mlp)
        )
        return (
            self.att_low * output_low
            + self.att_high * output_high
            + self.att_mlp * output_mlp
        ) * 3