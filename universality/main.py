import argparse
import time

import numpy as np
import torch
import torch_geometric as pyg

import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

from models import Convolution


def plot_losses(metrics, postfix):
    plt.figure(figsize=(7, 3))
    plt.rcParams.update({'font.size': 9})
    cm = plt.get_cmap('Set1')
    for i, key in enumerate(metrics):
        plt.plot(metrics[key], label=key, c=cm(i))
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('Step')
    plt.ylabel('MSE')
    plt.savefig(f'./figures/losses_{postfix}.svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Function approximation.')
    parser.add_argument('--runs', type=int, default=5, help='Number of nodes')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--conv_types', type=str, default='ACM', help='Convolution types')

    args = parser.parse_args()


    device = 'cpu'#('cuda' if torch.cuda.is_available() else 'cpu')
    n = 64
    d = 16
    k = 4
    n_layers = args.n_layers
    runs = args.runs
    metrics = {}
    for conv_type in args.conv_types.split(','):
        best_loss = 1e10
        best_losses = []

        for lr in [0.003]:#,0.001]:
            start = time.time()

            min_losses = []
            run_losses = []
            for i in range(runs):
                pyg.seed_everything(i)
                X = torch.randn((n,d), requires_grad=False).to(device)
                Y = torch.randn((n,d), requires_grad=False).to(device)
                connected = False
                while not connected:
                    edge_index = pyg.utils.erdos_renyi_graph(n,0.1)
                    connected = not pyg.utils.contains_isolated_nodes(edge_index)

                edge_index = edge_index.to(device)

                model = Convolution(conv_type, n, d, k, n_layers).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.9995)
                losses = []
                for i in range(40000):
                    out = model(X, edge_index)
                    loss = torch.nn.functional.l1_loss(out, Y)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    losses.append(loss.item())
                min_losses.append(np.min(losses))
                run_losses.append(losses)
            if np.mean(min_losses) < best_loss:
                best_loss = np.mean(min_losses)
                best_losses = np.mean(np.array(run_losses), axis=0)
            print(f'Avg: ({n},{d},{k},{conv_type},{n_layers};{lr})', np.mean(min_losses), np.std(min_losses), time.time()-start)
        metrics[f'{conv_type}'] = best_losses
    plot_losses(metrics, f'{args.runs}_{args.n_layers}')


