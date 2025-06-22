import torch
import torch_geometric as pyg

from methods import MCGC, LMGC, FAGCN, ACM


class Convolution(torch.nn.Module):
    def __init__(self, conv_type, n, d, k=4, n_layers=1):
        super(Convolution, self).__init__()
        self.conv_type = conv_type
        self.convs = torch.nn.ModuleList()
        self.enc = None
        if conv_type == 'GATv2':
            for i in range(n_layers):
                self.convs.append(pyg.nn.GATv2Conv(d,d, heads=k, concat=False, add_self_loops=False))
        elif conv_type == 'MCGC':
            self.convs.append(MCGC.MCGC(n, d))
        elif conv_type == 'FAGCN':
            for i in range(n_layers):
                self.convs.append(FAGCN.FAGCN(d, d, add_self_loops=False))
        elif conv_type == 'LMGC':
            self.convs.append(LMGC.LMGC(d, d, heads=k))
        elif conv_type == 'ACM':
            self.convs.append(ACM.ACM(d, d))
        elif conv_type == 'GIN':
            self.enc = torch.nn.Sequential(
                    torch.nn.Linear(d,d),
                    torch.nn.ReLU(),
                    torch.nn.Linear(d,d),
                    torch.nn.ReLU(),
                    torch.nn.Linear(d, d)
                )
            for i in range(n_layers):
                self.convs.append(pyg.nn.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(d,d),
                    torch.nn.ReLU(),
                    torch.nn.Linear(d,d),
                    torch.nn.ReLU(),
                    torch.nn.Linear(d, d)
                )))
        else:
            raise ValueError(f"Unknown convolution type {conv_type}")

    def forward(self, x, edge_index):
        if self.enc is not None:
            x = self.enc(x)
        for i, layer in enumerate(self.convs):
            x = layer(x, edge_index)
        return x