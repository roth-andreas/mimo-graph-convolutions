# What can we learn from MIMO graph convolutions?

This repository contains the code for the paper "What can we learn from MIMO graph convolutions?".

## Reference Implementation

* LMGC: See `./universality/LMGC.py` for a reference implemention.

## To reproduce the results in the paper

### Python environment setup with Conda

```bash
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

conda clean --all
```

### Reproduce results on function approximation
See `./universality/` for all details.

### Reproduce results on ZINC and ZINC12k
See `./lrgb/` for all details.

### Reproduce results on the heterophilic node classification tasks
See `./heterophilic_node/` for all details.

