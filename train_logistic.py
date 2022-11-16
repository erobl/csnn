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
from glob import glob

from utils.dataset import PointsetDataset
from utils.trainer import dd_initialize, train_initialize, train_head, finetune, load_evaluate, train_initialize_loaded
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

os.makedirs("experiment/%s/" % config['experiment_name'], exist_ok=True)
with open('experiment/%s/results.csv' % config['experiment_name'], 'w') as f:
    writer = csv.writer(f)
    row_names = ["seed", "hash"]
    writer.writerow(row_names + keys + [name + "_train" for name, _ in metrics] + [name + "_valid" for name, _ in metrics])

param_grid = ParameterGrid(config['param_grid'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for seed in range(n_seeds):
    dataset_train = PointsetDataset(config['datasets_train'], random_state=seed, **config['dataset_config'])
    dataset_valid = PointsetDataset(config['datasets_valid'], random_state=seed, **config['dataset_config'])

    if 'load_dd' not in config:
        diff, allpos_sample, allneg_sample, (score_pos, score_neg) = dd_initialize(dataset_train, dataset_valid, **config['experiment_config'], return_scores=True)
    else:
        print("Loading dd from past experiment: %s" % config['load_dd'])
    for params in tqdm(param_grid):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        paramsjson = json.dumps(params).encode('utf-8')
        param_hash = hashlib.md5(paramsjson).hexdigest()

        resdir = "experiment/%s/models/seed_%d/hash_%s" % (config['experiment_name'], seed, param_hash)
        alpha = params['alpha']
        if 'load_dd' in config:
            load_experiment = "experiment/%s/models/seed_%d/hash_*" % (config['load_dd'], seed)
            search_results = glob(load_experiment)
            assert len(search_results) == 1
            loaddir = search_results[0]
        else:
            if config['experiment_config']['bayesian_dd']:
                diff = (1 - np.exp(np.log(1 - alpha) + score_neg - score_pos))
            else:
                diff = (1/alpha) * np.exp(score_pos) - ((1-alpha)/alpha) * np.exp(score_neg)

        os.makedirs(resdir, exist_ok=True)

        if True:
            kwargs = params.copy()
            for key in config['experiment_config']:
                kwargs[key] = config['experiment_config'][key]

            if 'load_dd' in config:
                Xinit_train, yinit_train = torch.load("%s/dd_init.pt" % loaddir)
                model, Xinit_train, yinit_train = train_initialize_loaded(Xinit_train, yinit_train, **kwargs)
            else:
                model, Xinit_train, yinit_train = train_initialize(dataset_train, dataset_valid, diff, allpos_sample, allneg_sample, **kwargs)

            torch.save((Xinit_train, yinit_train), "%s/dd_init.pt" % resdir)
            torch.save(model.cpu(), "%s/model_init.pt" % resdir)

            model = train_head(dataset_train, dataset_valid, model, **config['experiment_config'])

            torch.save(model.cpu(), "%s/model_head.pt" % resdir)

            model, (preds_val, y_valid), (preds_train, y_train) = finetune(dataset_train, dataset_valid, model, **kwargs)

            torch.save(model, "%s/model.pt" % resdir)
        else:
            model, (preds_val, y_valid), (preds_train, y_train) = load_evaluate(dataset_train, dataset_valid, "%s/model.pt")

        train_metrics = []
        for name, metric in metrics:
            if name in ["accuracy", "f1", "precision", "recall"]:
                train_metrics.append(metric(y_train.numpy(), (preds_train > 0.5).to(int).numpy()))
            else:
                train_metrics.append(metric(y_train.numpy(), preds_train.numpy()))

        val_metrics = []
        for name, metric in metrics:
            if name in ["accuracy", "f1", "precision", "recall"]:
                val_metrics.append(metric(y_valid.numpy(), (preds_val > 0.5).to(int).numpy()))
            else:
                val_metrics.append(metric(y_valid.numpy(), preds_val.numpy()))


        with open('experiment/%s/results.csv' % config['experiment_name'], 'a') as f:
            writer = csv.writer(f)
            metadata = [seed, param_hash]
            param_list = [params[k] for k in keys]
            writer.writerow(metadata + param_list + train_metrics + val_metrics)
