import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import random

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dd_initialize(dataset_train, dataset_valid, max_iter=int(1e6), tol=1e-7):
    return None, None, None


def train_initialize(dataset_train, dataset_valid, max_iter=int(1e6), tol=1e-7,
                     lr=1e-5, architecture=[3,3], head_arch=[3,1], dropout=None,
                     batch_size=None, scheduler=False):
    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y


    from utils.PointSetCNN import PointSetCNN

    model = PointSetCNN(x_train.shape[-1], architecture=architecture, head_arch=head_arch, dropout_rate=dropout)
    model.to(device)

    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y

    x_train = x_train.to(device).float()
    x_valid = x_valid.to(device).float()
    y_train = y_train.to(device).float()
    y_valid = y_valid.to(device).float()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)

    num_epochs = max_iter
    last_loss = float('inf')
    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=250,
                T_mult=1,
        )
    for epoch in range(num_epochs):
        with torch.set_grad_enabled(True):
            model.train()
            if batch_size is None:
                optimizer.zero_grad()
                preds, leaf_probs = model(x_train, return_leaf_probs = True)
                loss = criterion(preds, y_train)

                loss.backward()
                optimizer.step()

                tr_loss = loss.detach().item()
                tr_acc = ((preds > 0.5) == y_train).float().mean().item()
            else:
                n_sampled = 0
                perm = torch.randperm(y_train.shape[0])
                tr_accs = []
                tr_losses = []
                while y_train.shape[0] > n_sampled:
                    optimizer.zero_grad()
                    preds, leaf_probs = model(x_train[perm[n_sampled:min(y_train.shape[0], n_sampled+batch_size)]], return_leaf_probs = True)
                    loss = criterion(preds, y_train[perm[n_sampled:min(y_train.shape[0], n_sampled+batch_size)]])

                    loss.backward()
                    optimizer.step()
                    tr_acc = ((preds > 0.5) == y_train[perm[n_sampled:min(y_train.shape[0], n_sampled+batch_size)]]).float().mean().item()
                    tr_accs.append(tr_acc)
                    tr_losses.append(loss.detach().item())
                    n_sampled += batch_size
                    # scheduler.step(epoch=epoch + n_sampled / y_train.shape[0])
                if scheduler:
                    scheduler.step()

                tr_loss = sum(tr_losses)
                tr_acc = sum(tr_accs)/len(tr_accs)

        if epoch % 250 == 0:
            with torch.set_grad_enabled(False):
                preds = model(x_valid)
                loss = criterion(preds, y_valid)
                te_loss = loss.item()
                te_acc = ((preds > 0.5) == y_valid).float().mean().item()

            print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f, te_loss: %.04f, te_acc: %.04f" % (epoch, tr_loss, tr_acc, te_loss, te_acc))

        if np.abs(tr_loss - last_loss) < tol:
            print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f, te_loss: %.04f, te_acc: %.04f" % (epoch, tr_loss, tr_acc, te_loss, te_acc))
            break
        last_loss = tr_loss

    with torch.set_grad_enabled(False):
        preds_val = model(x_valid)
        preds_train = model(x_train)
    return model.cpu(), (preds_val.cpu(), y_valid.cpu()), (preds_train.cpu(), y_train.cpu())

def load_evaluate(dataset_train, dataset_valid, fn):
    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y

    x_train = x_train.to(device).float()
    x_valid = x_valid.to(device).float()
    y_train = y_train.to(device).float()
    y_valid = y_valid.to(device).float()

    model = torch.load(fn)
    model = model.to(device)
    model.eval()

    with torch.set_grad_enabled(False):
        preds_val = model(x_valid)
        preds_train = model(x_train)

    return model.cpu(), (preds_val.cpu(), y_valid.cpu()), (preds_train.cpu(), y_train.cpu())
