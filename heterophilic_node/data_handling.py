from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
import torch
import numpy as np

from large_datasets import load_nc_dataset
import torch_geometric as pyg

def get_data(name, split=0):
  path = './data/' +name
  if name in ['chameleon','squirrel']:
    dataset = WikipediaNetwork(root=path, name=name)
  elif name in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(path ,name=name)
  elif name == 'film':
    dataset = Actor(root=path)
  elif name in ['Cora', 'Citeseer', 'Pubmed']:
    dataset = Planetoid(root=path, name=name, split='public')
  else:
    dataset = load_nc_dataset(name)

  data = dataset[0]
  if not name in ['Cora', 'Citeseer', 'Pubmed']:
    if name == 'arxiv-year':
      data = pyg.data.Data(data[0]['node_feat'], data[0]['edge_index'],y=data[1])
      split_idx = dataset.get_idx_split()
      train_mask, val_mask, test_mask = split_idx["train"], split_idx["valid"], split_idx["test"]

    else:
      if name in ['chameleon', 'squirrel']:
        splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
      if name in ['cornell', 'texas', 'wisconsin']:
        splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
      if name == 'film':
        splits_file = np.load(f'{path}/raw/{name}_split_0.6_0.2_{split}.npz')
      #if name in ['Cora', 'Citeseer', 'Pubmed']:
      #    splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
      train_mask = torch.tensor(splits_file['train_mask'], dtype=torch.bool)
      val_mask = torch.tensor(splits_file['val_mask'], dtype=torch.bool)
      test_mask = torch.tensor(splits_file['test_mask'], dtype=torch.bool)

    data.train_mask = train_mask.clone().detach().bool()
    data.val_mask = val_mask.clone().detach().bool()
    data.test_mask = test_mask.clone().detach().bool()

  return data
