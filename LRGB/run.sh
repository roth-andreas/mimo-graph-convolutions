#!/usr/bin/zsh

# Experiments on ZINC
for seed in {0,1,2,3}; do
  for lr in {0.001,0.0003,0.0001}; do
    for layers in {6,8,10}; do
      for name in {ZINC-GATv2,ZINC-GIN,ZINC-FAGCN,ZINC-LMGC}; do
        config=configs/${name}.yaml
        args="--cfg $config --repeat 1 gnn.self_loops True optim.num_warmup_epochs 0 optim.max_epoch 200 gnn.max_params 100000 gnn.residual False gnn.batchnorm False dataset.name full gnn.layers_post_mp 2 gnn.dim_inner 600 gnn.heads 4 gnn.keep_size False optim.base_lr $lr gnn.dropout 0.0 gnn.layers_mp $layers gnn.share_init False seed $seed name_tag ${layers}_${lr}"
        python main.py ${args}
      done
    done
  done
done

# Experiments on ZINC12k
for seed in {0,1,2,3}; do
  for lr in {0.001,0.0003,0.0001}; do
    for layers in {6,8,10}; do
      for name in {ZINC-GATv2,ZINC-GIN,ZINC-FAGCN,ZINC-LMGC}; do
        config=configs/${name}.yaml
        # Base
        args="--cfg $config --repeat 1 gnn.self_loops True optim.num_warmup_epochs 0 optim.max_epoch 1000 gnn.max_params 100000 gnn.residual False gnn.batchnorm False dataset.name subset gnn.layers_post_mp 2 gnn.dim_inner 600 gnn.heads 4 gnn.keep_size False optim.base_lr $lr gnn.dropout 0.0 gnn.layers_mp $layers gnn.share_init False seed $seed name_tag ${layers}_${lr}"
        python main.py ${args}
        # Base + Res
        args="--cfg $config --repeat 1 gnn.self_loops True optim.num_warmup_epochs 0 optim.max_epoch 1000 gnn.max_params 100000 gnn.residual True gnn.batchnorm False dataset.name subset gnn.layers_post_mp 2 gnn.dim_inner 600 gnn.heads 4 gnn.keep_size False optim.base_lr $lr gnn.dropout 0.0 gnn.layers_mp $layers gnn.share_init False seed $seed name_tag res_${layers}_${lr}"
        python main.py ${args}
        # Base + LapPE + JK + Res
        args="--cfg $config --repeat 1 gnn.self_loops True gnn.jk max optim.num_warmup_epochs 0 optim.max_epoch 1000 gnn.max_params 100000 gnn.residual True dataset.node_encoder_name TypeDictNode+LapPE posenc_LapPE.enable True gnn.batchnorm False dataset.name subset gnn.layers_post_mp 2 gnn.dim_inner 600 gnn.heads 4 gnn.keep_size False optim.base_lr $lr gnn.dropout 0.0 gnn.layers_mp $layers gnn.share_init False seed $seed name_tag full_${layers}_${lr}"
        python main.py ${args}
        # Base + LapPE
        args="--cfg $config --repeat 1 gnn.self_loops True optim.num_warmup_epochs 0 optim.max_epoch 1000 gnn.max_params 100000 gnn.residual False dataset.node_encoder_name TypeDictNode+LapPE posenc_LapPE.enable True gnn.batchnorm False dataset.name subset gnn.layers_post_mp 2 gnn.dim_inner 600 gnn.heads 4 gnn.keep_size False optim.base_lr $lr gnn.dropout 0.0 gnn.layers_mp $layers gnn.share_init False seed $seed name_tag lappe_${layers}_${lr}"
        python main.py ${args}
        # Base + JK
        args="--cfg $config --repeat 1 gnn.self_loops True gnn.jk max optim.num_warmup_epochs 0 optim.max_epoch 1000 gnn.max_params 100000 gnn.residual False gnn.batchnorm False dataset.name subset gnn.layers_post_mp 2 gnn.dim_inner 600 gnn.heads 4 gnn.keep_size False optim.base_lr $lr gnn.dropout 0.0 gnn.layers_mp $layers gnn.share_init False seed $seed name_tag jkmax_${layers}_${lr}"
        python main.py ${args}
      done
    done
  done
done