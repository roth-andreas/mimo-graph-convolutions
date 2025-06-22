import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)


class FAGCN(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_l = Linear(in_channels, out_channels, bias=bias,
                                weight_initializer='glorot')

        self.mlp = torch.nn.Sequential(torch.nn.Linear(2*in_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Tensor:

        x_l = self.lin_l(x)

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value,
                num_nodes=num_nodes)

        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, None, x.size(0),
            False, False, self.flow, x.dtype)

        alpha = torch.tanh(self.mlp(torch.cat((x[edge_index[1]],x[edge_index[0]]),dim=1))) * edge_weight.unsqueeze(dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)


        # propagate_type: (x: PairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x_l, alpha=alpha)

        out = out.mean(dim=1)

        return out

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)