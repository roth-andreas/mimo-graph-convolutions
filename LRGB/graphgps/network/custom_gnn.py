import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from graphgps.layer.dag_utils import init_graph
import torch_geometric as pyg

from graphgps.layer.gin_conv_layer import GINConvLayer
from graphgps.layer.gatv2_conv_layer import GATv2ConvLayer
from graphgps.layer.fagcn_conv_layer import FAGCNConvLayer
from graphgps.layer.lmg_conv_layer import LMGConvLayer
from graphgps.layer.acm_conv_layer import ACMConvLayer
import time
import numpy as np

@register_network('custom_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.model_type = cfg.gnn.layer_type
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        self.total_times = []
        self.ordering_times = []
        self.split_times = []
        self.transformation_times = []
        self.aggregation_times = []

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        self.gnn_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        for _ in range(cfg.gnn.layers_mp):
            self.gnn_layers.append(conv_model(dim_in,
                                              dim_in,
                                              dropout=cfg.gnn.dropout,
                                              residual=cfg.gnn.residual))
            if cfg.gnn.batchnorm:            
                self.norm_layers.append(pyg.nn.norm.PairNorm(scale_individually=True))
        GNNHead = register.head_dict[cfg.gnn.head]
        if cfg.gnn.jk is not None:
            self.jk = JumpingKnowledge(cfg.gnn.jk)
            jk_dim = cfg.gnn.dim_inner if cfg.gnn.jk == 'max' else cfg.gnn.dim_inner * cfg.gnn.layers_mp
            self.post_mp = GNNHead(dim_in=jk_dim, dim_out=dim_out)
        else:
            self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(cfg.gnn.dropout),
        )

    def build_conv_model(self, model_type):
        if model_type == 'ginconv':
            return GINConvLayer
        elif model_type == 'gatv2conv':
            return GATv2ConvLayer
        elif model_type == 'lmgconv':
            return LMGConvLayer
        elif model_type == 'fagcnconv':
            return FAGCNConvLayer
        elif model_type == 'acmconv':
            return ACMConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)

        xs = []
        for idx, conv in enumerate(self.gnn_layers):
            batch = conv(batch)
            if len(self.norm_layers) > 0:
                batch.x = self.norm_layers[idx](batch.x, batch.batch)
            xs.append(batch.x)
        if cfg.gnn.jk:
            batch.x = self.jk(xs)

        batch = self.post_mp(batch)
        return batch
