import torch
import numpy as np
import random
import torch
import csv
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
from yaml import load, FullLoader
import sys
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.dataset import PointsetDataset
from utils.trainer import dd_initialize, train_initialize, train_head, finetune, load_evaluate
from utils.DensityDifference import DensityDifference
import json
import hashlib

with open(sys.argv[1], 'r') as f:
    config = load(f, Loader=FullLoader)

n_folds = config['n_folds']
n_seeds = config['n_seeds']
prop_folds = 1-1/n_folds

metrics = [
    ("accuracy", accuracy_score),
    ("f1", f1_score),
    ("precision", precision_score),
    ("recall", recall_score),
    ("auc", roc_auc_score),
]

print(config['param_grid'])
param_grid = ParameterGrid(config['param_grid'])
sample = param_grid[0]

keys = list(sample.keys())

config['experiment_name'] += "_dd"

os.makedirs("ablation_experiment/%s/" % config['experiment_name'], exist_ok=True)
with open('ablation_experiment/%s/results.csv' % config['experiment_name'], 'w') as f:
    writer = csv.writer(f)
    row_names = ["seed", "hash"]
    writer.writerow(row_names + keys + [name + "_train" for name, _ in metrics] + [name + "_valid" for name, _ in metrics])

param_grid = ParameterGrid(config['param_grid'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for seed in range(n_seeds):
    dataset_train = PointsetDataset(config['datasets_train'], random_state=seed, **config['dataset_config'])
    dataset_valid = PointsetDataset(config['datasets_valid'], random_state=seed, **config['dataset_config'])

    #diff, allpos_sample, allneg_sample, (score_pos, score_neg) = dd_initialize(dataset_train, dataset_valid, **config['experiment_config'], return_scores=True)
    for params in tqdm(param_grid):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        paramsjson = json.dumps(params).encode('utf-8')
        param_hash = hashlib.md5(paramsjson).hexdigest()

        resdir = "ablation_experiment/%s/models/seed_%d/hash_%s" % (config['experiment_name'], seed, param_hash)

        alpha = params['alpha']
        dd = DensityDifference(alpha, sample_size=config['experiment_config']['dd_sample_size'])
        allpos_sample, allneg_sample = dd.fit(dataset_train.X, dataset_train.y)
        # diff = (1/alpha) * np.exp(score_pos) - ((1-alpha)/alpha) * np.exp(score_neg)
        # diff =  1 - np.exp(np.log(1 - alpha) + score_neg - score_pos)

        os.makedirs(resdir, exist_ok=True)

        def score(X):
            perm = np.random.permutation(X.shape[0])[:dd.sample_size]
            diff = dd.score_samples(X[perm])
            return diff.clip(0,1).mean()

        preds_train = Parallel(n_jobs=10)(delayed(score)(X) for X in tqdm(dataset_train.X))
        preds_val = Parallel(n_jobs=10)(delayed(score)(X) for X in tqdm(dataset_valid.X))
        preds_train = np.asarray(preds_train)
        preds_val = np.asarray(preds_val)

        y_train = dataset_train.y
        y_valid = dataset_valid.y

        train_metrics = []
        for name, metric in metrics:
            if name in ["accuracy", "f1", "precision", "recall"]:
                train_metrics.append(metric(y_train.numpy(), (preds_train > 0.5).astype(int)))
            else:
                train_metrics.append(metric(y_train.numpy(), preds_train))

        val_metrics = []
        for name, metric in metrics:
            if name in ["accuracy", "f1", "precision", "recall"]:
                val_metrics.append(metric(y_valid.numpy(), (preds_val > 0.5).astype(int)))
            else:
                val_metrics.append(metric(y_valid.numpy(), preds_val))


        with open('ablation_experiment/%s/results.csv' % config['experiment_name'], 'a') as f:
            writer = csv.writer(f)
            metadata = [seed, param_hash]
            param_list = [params[k] for k in keys]
            writer.writerow(metadata + param_list + train_metrics + val_metrics)
