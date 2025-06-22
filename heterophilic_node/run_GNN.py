import copy
import time

from models import *
import torch
import torch.optim as optim
import numpy as np
from data_handling import get_data
import argparse


def train(args, split, verbose):
    data = get_data(args.dataset, split)

    best_eval_acc = 0
    best_eval_loss = 1e5
    bad_counter = 0
    best_test_acc = 0
    best_train_loss = 1e5
    not_improved = 0
    best_epoch = 0

    nout = torch.max(data.y)+1

    lower = False
    max_params = 100000
    while not lower:
        model = SimpleModel(data.num_node_features, args.nhid, nout, args.nlayers, args.conv_type, args.heads,
                            args.drop).to(args.device)
        num_params = pyg.graphgym.utils.comp_budget.params_count(model)
        lower = num_params <= max_params
        if not lower:
            args.nhid -= 1



    lf = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    @torch.no_grad()
    def test(model, data):
        model.eval()
        logits, accs, losses = model(data)[0], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            loss = lf(logits[mask], data.y.squeeze()[mask])
            pred = logits[mask].max(1)[1]#.cpu()
            acc = pred.eq(data.y[mask].squeeze()).sum().item() / mask.sum().item()
            accs.append(acc)
            losses.append(loss.item())
        return accs, losses

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out, loss_aux = model(data.to(args.device))
        loss = lf(out[data.train_mask], data.y.squeeze()[data.train_mask]) + loss_aux
        loss.backward()
        optimizer.step()

        [train_acc, val_acc, test_acc], [train_loss, val_loss, test_loss] = test(model, data)

        if args.use_val_acc == True:
            if (val_acc > best_eval_acc):
                best_eval_acc = val_acc
                best_test_acc = test_acc
                best_eval_loss = val_loss
                best_epoch = epoch
                #if verbose:
                #    print(f"{epoch}: Train loss: {train_loss:.3f}, Acc: {train_acc:.3f}, Val loss: {val_loss:.3f}, Acc: {val_acc:.3f}, Test: loss: {test_loss:.3f}, acc: {test_acc:.3f}")
                not_improved = 0
            else:
                not_improved += 1

        else:
            if (val_loss < best_eval_loss):
                best_eval_loss = val_loss
                best_test_acc = test_acc
                best_eval_acc = val_acc
                best_epoch = epoch
                if verbose:
                    print(f"Best loss: {val_loss:.3f}, Acc: {val_acc:.3f}, Test: loss: {test_loss:.3f}, acc: {test_acc:.3f}")
                not_improved = 0
            else:
                not_improved += 1

        if train_loss > best_train_loss:
            bad_counter += 1
        else:
            best_train_loss = train_loss
            bad_counter = 0

        if ((bad_counter+1) == 20):
        #    #break
            for g in optimizer.param_groups:
                g['lr'] = max(g['lr'] / 3, 0.0001)
                #if verbose:
                #    print(f"Reduced LR to {g['lr']}!")
            bad_counter = 0
            best_train_loss = train_loss

        if (not_improved == args.patience):
            break

        #if verbose:
        #    print(f'Split: {split:01d}, Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Loss: {train_loss:.4f},{val_loss:.4f}')
    if verbose:
        print(f'Epoch {best_epoch}: Final Train Loss: {train_loss:.3f}, Val Loss: {best_eval_loss:.3f}, Test Acc {best_test_acc:.3f}')#, Baseline: {torch.max(torch.bincount(data.y[data.test_mask])) / len(data.y[data.test_mask]):.3f}')

    return best_eval_acc, best_eval_loss, best_test_acc, best_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--dataset', type=str, default='squirrel',
                        help='dataset name: texas, wisconsin, film, squirrel, chameleon, cornell')
    parser.add_argument('--conv_type', type=str, default='lmgc',
                        help='base GNN model used with G^2: GraphSAGE, GCN, GAT')
    parser.add_argument('--nhid', type=int, default=128 ,
                        help='number of hidden node features')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='max epochs')
    parser.add_argument('--patience', type=int, default=200,
                        help='patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--drop_in', type=float, default=0.5,
                        help='input dropout rate')
    parser.add_argument('--drop', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.00,
                        help='weight_decay')
    parser.add_argument('--G2_exp', type=float, default=2.5,
                        help='exponent p in G^2')
    parser.add_argument('--use_val_acc', type=bool, default=True,
                        help='use validation accuracy for early stoppping -- otherwise use validation loss')
    parser.add_argument('--use_G2_conv', type=bool, default=False,
                        help='use a different GNN model for the gradient gating method')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='computing device')
    parser.add_argument('--heads', type=int, default=4)

    args = parser.parse_args()

    print(f'Using cuda: {torch.cuda.is_available()}')
    args.use_val_acc = True
    verbose = True
    # Hyperparameter tuning
    best_val_acc = 0
    best_val_loss = np.inf
    best_test = []
    best_epochs = []
    best_args = None
    for num_layers in [2]:
        head_list = [4] if args.conv_type in ['lmgc','gatv2'] else [1]
        for heads in head_list:
            for h_dim in [100]:
                for drop in [0.0,0.25,0.5]:
                    for lr in [0.01,0.003,0.001]:
                        try:
                            args.lr = lr
                            args.drop_in = drop#_in
                            args.drop = drop
                            args.nhid = h_dim
                            args.heads = heads
                            args.nlayers = num_layers
                            n_splits = 10
                            val_accs = []
                            val_losses = []
                            test_accs = []
                            epochs = []
                            start = time.time()
                            for split in range(n_splits):#[6]:#
                                val_acc, val_loss, test_acc, epoch = train(args, split, verbose)
                                test_accs.append(test_acc)
                                val_accs.append(val_acc)
                                val_losses.append(val_loss)
                                epochs.append(epoch)

                            log = f'({num_layers}-{h_dim}-{drop}-{lr}) Val Acc: {np.mean(val_accs):.4f}, Val Loss:{np.mean(val_losses):.4f}, Test Acc: {np.mean(test_accs):.4f} epochs: {np.mean(epochs):.1f}({time.time() - start})'
                            print(log)

                            if np.mean(val_accs) > best_val_acc:# or (not args.use_val_acc and np.mean(val_losses) < best_val_loss):
                                best_val_acc = np.mean(val_accs)
                                best_val_loss = np.mean(val_losses)
                                best_test = test_accs
                                best_epochs = epochs
                                best_args = copy.deepcopy(args)
                                print("-"*10,f"New Best: {np.mean(best_test):.3f}+-{np.std(best_test):.3f}, {best_val_acc:.3f}, {best_val_loss:.3f}","-"*10)
                        except EOFError as e:
                            print("Out of memory!")
                            print(e)
                            continue

    #best_results = np.array(best_results)
    mean_acc = np.mean(best_test)
    std = np.std(best_test)

    log = f'({num_layers}-{args.heads}-{lr})Final test results -- mean: {mean_acc:.4f}, std: {std:.4f}, epochs: {np.mean(best_epochs):.1f}+-{np.std(best_epochs):.2f}'
    print(log)
    print(best_args)
    mean_test_accs = []
    for repeat in range(5):
        repeat_test_accs = []
        for split in range(n_splits):  # [6]:#
            val_acc, val_loss, test_acc, epoch = train(best_args, split, verbose)
            repeat_test_accs.append(test_acc)
        mean_test_accs.append(np.mean(repeat_test_accs))

    print(f'Final run: {np.mean(mean_test_accs):.3f}+-{np.std(mean_test_accs):.3f}')
