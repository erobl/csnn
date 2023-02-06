import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import random
import csv

from utils.DensityDifference import DensityDifference
from utils.dataset import PointsetDataset

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dd_initialize(dataset_train, dataset_valid, alpha=0.75, max_iter=int(1e6),
                  tol=1e-7, dd_sample_size=10000, dd_threshold=95,
                  return_scores=False, bayesian_dd=False, n_restarts=5):
    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y

    allpos = x_train[y_train == 1].reshape((-1, x_train.shape[-1])).numpy()
    allneg = x_train[y_train == 0].reshape((-1, x_train.shape[-1])).numpy()

    dd = DensityDifference(alpha, sample_size=dd_sample_size)

    allpos_sample, allneg_sample = dd.fit(x_train, y_train)

    if return_scores:
        diff, scores = dd.score_samples(allpos_sample, return_scores=return_scores)
    else:
        diff = dd.score_samples(allpos_sample, return_scores=return_scores)

    if return_scores:
        return diff, allpos_sample, allneg_sample, scores
    else:
        return diff, allpos_sample, allneg_sample


def train_initialize(dataset_train, dataset_valid, diff, allpos_sample, 
                     allneg_sample, alpha=0.75, max_iter=int(1e6), tol=1e-7, 
                     dd_sample_size=10000, dd_threshold=95,
                     lr=1e-5, architecture=[8,4,2,1], negative_penalty=1,
                     bayesian_dd=False, n_restarts=5):
    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y


    threshold = np.percentile(diff, dd_threshold)

    allpos_torch = torch.tensor(allpos_sample)
    allneg_torch = torch.tensor(allneg_sample)
    nt = (diff > threshold).sum()

    Xinit_train = torch.cat((allpos_torch[diff > threshold,:], allneg_torch[:nt,:]), 0).float().unsqueeze(1)
    yinit_train = torch.cat((torch.ones(nt,), torch.zeros(nt,)), 0).float()

    from utils.PointSetNN import PointSetNN

    Xinit_train = Xinit_train.to(device)
    yinit_train = yinit_train.to(device)

    # number of restarts for the model, it sometimes doesn't initialize correctly
    # therefore we need some restarts to get the correct initialization
    accs = []
    models = []
    print("Initializing models with %d restarts" % n_restarts)
    for i in range(n_restarts):
        # density-based initialization
        model = PointSetNN(Xinit_train.shape[-1], architecture)

        model = model.to(device)

        criterion = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        num_epochs = max_iter
        last_loss = float('inf')
        for epoch in range(num_epochs):
            with torch.set_grad_enabled(True):
                model.train()
                optimizer.zero_grad()
                _, leaf_probs = model(Xinit_train, return_leaf_probs = True)
                leaf_probs = leaf_probs.squeeze()
                crit_loss = criterion(leaf_probs, yinit_train)

                loss =  crit_loss

                tr_loss = loss.item()
                tr_acc = ((leaf_probs > 0.5) == yinit_train).float().mean().item()
                loss.backward()
                optimizer.step()

            if epoch % 250 == 0:
                print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f" % (epoch, tr_loss, tr_acc))
            if np.abs(tr_loss - last_loss) < tol:
                print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f" % (epoch, tr_loss, tr_acc))
                break
            last_loss = tr_loss

        with torch.set_grad_enabled(False):
            _, leaf_probs = model(Xinit_train, return_leaf_probs = True)
            leaf_probs = leaf_probs.squeeze()
            crit_loss = criterion(leaf_probs, yinit_train)

            loss =  crit_loss

            tr_loss = loss.item()
            tr_acc = ((leaf_probs > 0.5) == yinit_train).float().mean().item()

        accs.append(tr_acc)
        models.append(model.cpu())

    best_i = np.argmax(accs)

    print("Best accuracy: %.02f" % accs[best_i])
    model = models[best_i].to(device)

    return model, Xinit_train, yinit_train

def train_initialize_loaded(Xinit_train, yinit_train,
                     alpha=0.75, max_iter=int(1e6), tol=1e-7,
                     dd_sample_size=10000, dd_threshold=95,
                     lr=1e-5, architecture=[8,4,2,1], negative_penalty=1,
                     bayesian_dd=False, n_restarts=5):
    from utils.PointSetNN import PointSetNN

    Xinit_train = Xinit_train.to(device)
    yinit_train = yinit_train.to(device)

    # number of restarts for the model, it sometimes doesn't initialize correctly
    # therefore we need some restarts to get the correct initialization
    accs = []
    models = []
    print("Initializing models with %d restarts" % n_restarts)
    for i in range(n_restarts):
        # density-based initialization
        model = PointSetNN(Xinit_train.shape[-1], architecture)

        model = model.to(device)

        criterion = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        num_epochs = max_iter
        last_loss = float('inf')
        for epoch in range(num_epochs):
            with torch.set_grad_enabled(True):
                model.train()
                optimizer.zero_grad()
                _, leaf_probs = model(Xinit_train, return_leaf_probs = True)
                leaf_probs = leaf_probs.squeeze()
                crit_loss = criterion(leaf_probs, yinit_train)

                loss =  crit_loss

                tr_loss = loss.item()
                tr_acc = ((leaf_probs > 0.5) == yinit_train).float().mean().item()
                loss.backward()
                optimizer.step()

            if epoch % 250 == 0:
                print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f" % (epoch, tr_loss, tr_acc))
            if np.abs(tr_loss - last_loss) < tol:
                print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f" % (epoch, tr_loss, tr_acc))
                break
            last_loss = tr_loss

        with torch.set_grad_enabled(False):
            _, leaf_probs = model(Xinit_train, return_leaf_probs = True)
            leaf_probs = leaf_probs.squeeze()
            crit_loss = criterion(leaf_probs, yinit_train)

            loss =  crit_loss

            tr_loss = loss.item()
            tr_acc = ((leaf_probs > 0.5) == yinit_train).float().mean().item()

        accs.append(tr_acc)
        models.append(model.cpu())

    best_i = np.argmax(accs)

    print("Best accuracy: %.02f" % accs[best_i])
    model = models[best_i].to(device)

    return model, Xinit_train, yinit_train

def train_head(dataset_train, dataset_valid, model, alpha=0.75, 
               max_iter=int(1e6), tol=1e-7, dd_sample_size=10000, 
               dd_threshold=95, 
               lr=1e-5, architecture=[8,4,2,1], negative_penalty=1,
               bayesian_dd=False, n_restarts=5):
    model.to(device)

    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y

    x_train = x_train.to(device).float()
    x_valid = x_valid.to(device).float()
    y_train = y_train.to(device).float()
    y_valid = y_valid.to(device).float()

    # train just the "head" of the model
    _, leaf_probs = model(x_train, return_leaf_probs = True)
    embedded_vector = leaf_probs.detach().mean(axis=1)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-2, weight_decay=1e-1)
    num_epochs = max_iter
    last_loss = float('inf')
    for epoch in range(num_epochs):
        with torch.set_grad_enabled(True):
            model.train()
            optimizer.zero_grad()
            preds = model.head(embedded_vector)
            preds = preds.squeeze(1)

            loss = criterion(preds, y_train)

            loss.backward()
            optimizer.step()

            tr_loss = loss.detach().item()
            tr_acc = ((preds > 0.5) == y_train).float().mean().item()

        if epoch % 250 == 0:
            with torch.set_grad_enabled(False):
                preds = model(x_valid)
                preds = preds.squeeze(1)
                loss = criterion(preds, y_valid)
                te_loss = loss.item()
                te_acc = ((preds > 0.5) == y_valid).float().mean().item()

            print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f, te_loss: %.04f, te_acc: %.04f" % (epoch, tr_loss, tr_acc, te_loss, te_acc))

        if np.abs(tr_loss - last_loss) < tol and epoch > 1000:
            print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f, te_loss: %.04f, te_acc: %.04f" % (epoch, tr_loss, tr_acc, te_loss, te_acc))
            break
        last_loss = tr_loss

    return model


def finetune(dataset_train, dataset_valid, model, alpha=0.75, max_iter=int(1e6),
             tol=1e-7, dd_sample_size=10000, dd_threshold=95,
             lr=1e-5, architecture=[8,4,2,1], negative_penalty=1, batch_size = None,
             bayesian_dd=False, n_restarts=5):
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
    for epoch in range(num_epochs):
        with torch.set_grad_enabled(True):
            model.train()

            if batch_size is None:
                optimizer.zero_grad()
                preds, leaf_probs = model(x_train, return_leaf_probs = True)
                preds = preds.squeeze(1)
                # positive_loss = criterion(preds[y_train == 1], y_train[y_train == 1])
                # negative_loss = criterion(preds[y_train != 1], y_train[y_train != 1])
                # crit_loss = negative_penalty * negative_loss + positive_loss

                loss =  criterion(preds, y_train)

                loss.backward()
                optimizer.step()
            else:
                n_sampled = 0
                perm = torch.randperm(y_train.shape[0])
                while y_train.shape[0] > n_sampled:
                    optimizer.zero_grad()
                    preds, leaf_probs = model(x_train[perm[n_sampled:n_sampled+batch_size]], return_leaf_probs = True)
                    preds = preds.squeeze(1)
                    # positive_loss = criterion(preds[y_train == 1], y_train[y_train == 1])
                    # negative_loss = criterion(preds[y_train != 1], y_train[y_train != 1])
                    # crit_loss = negative_penalty * negative_loss + positive_loss

                    loss =  criterion(preds, y_train[perm[n_sampled:n_sampled+batch_size]])

                    loss.backward()
                    optimizer.step()

                    n_sampled += batch_size

            tr_loss = loss.detach().item()
            tr_acc = ((preds > 0.5) == y_train).float().mean().item()

        if epoch % 250 == 0:
            with torch.set_grad_enabled(False):
                preds = model(x_valid)
                preds = preds.squeeze(1)
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
        preds_val = preds_val.squeeze(1)
        preds_train = model(x_train)
        preds_train = preds_train.squeeze(1)
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
        preds_val = preds_val.squeeze(1)
        preds_train = model(x_train)
        preds_train = preds_train.squeeze(1)

    return model.cpu(), (preds_val.cpu(), y_valid.cpu()), (preds_train.cpu(), y_train.cpu())
