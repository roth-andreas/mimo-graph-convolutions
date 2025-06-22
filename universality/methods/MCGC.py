import torch
import torch_geometric as pyg

class MCGC(torch.nn.Module):
    def __init__(self, n, d):
        super(MCGC, self).__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Linear(d,d,bias=False) for _ in range(n)])
        self.evecs = None

    def forward(self, x, edge_index):
        if self.evecs is None:
            laplacian = pyg.utils.get_laplacian(edge_index, normalization='sym')
            laplacian = pyg.utils.to_dense_adj(laplacian[0], edge_attr=laplacian[1])[0]
            self.evecs = torch.linalg.eigh(laplacian)[1]

        x_out = torch.zeros_like(x)
        for i, conv in enumerate(self.convs):
            x_out += self.evecs[:,i].view(-1,1) @ self.evecs[:,i].view(1,-1) @ conv(x)
        return x_out