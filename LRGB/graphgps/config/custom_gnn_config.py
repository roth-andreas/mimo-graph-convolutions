from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """
    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.heads = 4
    cfg.gnn.attn_dropout = 0.1

    cfg.gnn.use_vn = True
    cfg.gnn.vn_pooling = 'add'

    cfg.gnn.norm_type = None
    cfg.gnn.keep_size = False
    cfg.gnn.share_init = False
    cfg.gnn.self_loops = False
    cfg.gnn.jk = None
    cfg.gnn.max_params = 500000
    cfg.gnn.share_weights = True
