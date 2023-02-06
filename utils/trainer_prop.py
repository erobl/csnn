import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
import random
import csv

from utils.ProportionDensityDifference import ProportionDensityDifference

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dd_initialize(dataset_train, dataset_valid, max_iter=int(1e6),
                  tol=1e-7, dd_sample_size=10000, dd_threshold='auto',
                  bayesian_dd=False, n_jobs=32, n_restarts=5
                  ):
    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y
    alpha_train = dataset_train.proportion

    alpha_train[(y_train == 1) & (alpha_train == 0.0)] += 1e-12

    dd = ProportionDensityDifference(sample_size=dd_sample_size, n_jobs=n_jobs)

    allpos_sample, allneg_sample = dd.fit(x_train.numpy(), y_train.numpy(), alpha_train.numpy())

    diff = dd.score_samples(allpos_sample)

    return diff, allpos_sample, allneg_sample


def train_initialize(dataset_train, dataset_valid, diff, allpos_sample, 
                     allneg_sample, max_iter=int(1e6), tol=1e-7, 
                     dd_sample_size=10000, dd_threshold='auto',
                     lr=1e-5, architecture=[8,4,2,1], negative_penalty=1,
                     bayesian_dd=False, n_restarts=5):
    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y
    alpha_train = dataset_train.proportion


    if dd_threshold != 'auto' and not bayesian_dd:
        threshold = np.percentile(diff, dd_threshold)

        allpos_torch = torch.tensor(allpos_sample)
        allneg_torch = torch.tensor(allneg_sample)
        neg_perm = np.random.permutation(allneg_torch.shape[0])
        allneg_torch = allneg_torch[neg_perm]
        nt = (diff > threshold).sum()

        Xinit_train = torch.cat((allpos_torch[diff > threshold,:], allneg_torch[:nt,:]), 0).float().unsqueeze(1)
        yinit_train = torch.cat((torch.ones(nt,), torch.zeros(nt,)), 0).float()
    else:
        assert len(diff) == alpha_train[y_train==1].shape[0]
        assert len(diff) == allpos_sample.shape[0]
        nt = 0
        Xinit_pos = []
        for d, alpha, pos in zip(diff, alpha_train[y_train==1], allpos_sample):
            percentile = (1-alpha)*100
            threshold = np.percentile(d, percentile)

            nt += (d > threshold).sum()
            Xinit_pos.append(pos[d > threshold])
        Xinit_pos = torch.tensor(np.concatenate(Xinit_pos))
        allneg_torch = torch.tensor(allneg_sample)
        allneg_torch = allneg_torch.reshape(-1, allneg_torch.shape[-1])
        neg_perm = np.random.permutation(allneg_torch.shape[0])
        allneg_torch = allneg_torch[neg_perm]
        Xinit_train = torch.cat((Xinit_pos, allneg_torch[:nt,:]), 0).float().unsqueeze(1)
        yinit_train = torch.cat((torch.ones(nt,), torch.zeros(nt,)), 0).float()


    from utils.ProportionPointSetNN import ProportionPointSetNN

    Xinit_train = Xinit_train.to(device)
    yinit_train = yinit_train.to(device)

    # number of restarts for the model, it sometimes doesn't initialize correctly
    # therefore we need some restarts to get the correct initialization
    accs = []
    models = []
    print("Initializing models with %d restarts" % n_restarts)
    for i in range(n_restarts):
        # density-based initialization
        model = ProportionPointSetNN(Xinit_train.shape[-1], architecture)

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
                loss = criterion(leaf_probs, yinit_train)

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
                     max_iter=int(1e6), tol=1e-7, 
                     dd_sample_size=10000, dd_threshold='auto',
                     lr=1e-5, architecture=[8,4,2,1], negative_penalty=1,
                     bayesian_dd=False, n_restarts=5):
    from utils.ProportionPointSetNN import ProportionPointSetNN

    Xinit_train = Xinit_train.to(device)
    yinit_train = yinit_train.to(device)

    # number of restarts for the model, it sometimes doesn't initialize correctly
    # therefore we need some restarts to get the correct initialization
    accs = []
    models = []
    print("Initializing models with %d restarts" % n_restarts)
    for i in range(n_restarts):
        # density-based initialization
        model = ProportionPointSetNN(Xinit_train.shape[-1], architecture)

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
                loss = criterion(leaf_probs, yinit_train)

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

def finetune(dataset_train, dataset_valid, model, max_iter=int(1e6),
             tol=1e-7, dd_sample_size=10000, dd_threshold='auto',
             lr=1e-5, architecture=[8,4,2,1], negative_penalty=1,
             bayesian_dd=False, n_restarts=5):
    model.to(device)

    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y
    alpha_train = dataset_train.proportion

    x_train = x_train.to(device).float()
    x_valid = x_valid.to(device).float()
    y_train = y_train.to(device).float()
    y_valid = y_valid.to(device).float()
    alpha_train = alpha_train.to(device).float()

    criterion = torch.nn.MSELoss(reduction='sum')
    criterion_mean = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-1)

    num_epochs = max_iter
    last_loss = float('inf')
    for epoch in range(num_epochs):
        with torch.set_grad_enabled(True):
            model.train()
            optimizer.zero_grad()
            preds, leaf_probs = model(x_train, return_leaf_probs = True)
            preds = preds.squeeze(1)
            positive_loss = criterion(preds[y_train == 1], alpha_train[y_train == 1])
            negative_loss = criterion(preds[y_train != 1], alpha_train[y_train != 1])
            crit_loss = negative_penalty * negative_loss + positive_loss

            loss =  crit_loss

            loss.backward()
            optimizer.step()

            tr_loss = loss.detach().item()
            tr_acc = (1 - criterion_mean(preds, alpha_train)).float().mean().item()

        if epoch % 250 == 0:
            with torch.set_grad_enabled(False):
                preds = model(x_valid)
                preds = preds.squeeze(1)
                mean_positive = preds[y_valid == 1].mean()
                mean_negative = preds[y_valid != 1].mean()
                te_acc_3 = ((preds > 0.03) == y_valid).float().mean().item()
                te_acc_1 = ((preds > 0.01) == y_valid).float().mean().item()
                te_acc_05 = ((preds > 0.005) == y_valid).float().mean().item()

            print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f, mean_positive: %.04f, mean_negative: %.04f, accuracy: 0.03 %.04f; 0.01 %.04f; 0.005 %.04f" % (epoch, tr_loss, tr_acc, mean_positive, mean_negative, te_acc_3, te_acc_1, te_acc_05))

        if np.abs(tr_loss - last_loss) < tol:
            print("[Epoch %d] tr_loss: %.04f, tr_acc: %.04f, mean_positive: %.04f, mean_negative: %.04f, accuracy: 0.03 %.04f; 0.01 %.04f; 0.005 %.04f" % (epoch, tr_loss, tr_acc, mean_positive, mean_negative, te_acc_3, te_acc_1, te_acc_05))
            break
        last_loss = tr_loss

    with torch.set_grad_enabled(False):
        preds_val = model(x_valid)
        preds_val = preds_val.squeeze(1)
        preds_train = model(x_train)
        preds_train = preds_train.squeeze(1)
    return model.cpu(), (preds_val.cpu(), y_valid.cpu()), (preds_train.cpu(), y_train.cpu(), alpha_train.cpu())

def load_evaluate(dataset_train, dataset_valid, fn):
    x_train = dataset_train.X
    x_valid = dataset_valid.X
    y_train = dataset_train.y
    y_valid = dataset_valid.y
    alpha_train = dataset_train.proportion

    x_train = x_train.to(device).float()
    x_valid = x_valid.to(device).float()
    y_train = y_train.to(device).float()
    y_valid = y_valid.to(device).float()
    alpha_train = alpha_train.to(device).float()

    model = torch.load(fn)
    model = model.to(device)
    model.eval()

    with torch.set_grad_enabled(False):
        preds_val = model(x_valid)
        preds_val = preds_val.squeeze(1)
        preds_train = model(x_train)
        preds_train = preds_train.squeeze(1)

    return model.cpu(), (preds_val.cpu(), y_valid.cpu()), (preds_train.cpu(), y_train.cpu(), alpha_train.cpu())
