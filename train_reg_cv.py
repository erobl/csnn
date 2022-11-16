import torch
import numpy as np
import random
import torch
import csv
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from yaml import load, FullLoader
import sys
from tqdm import tqdm
import json
import hashlib

from utils.dataset import PointsetDataset
from utils.trainer_prop import dd_initialize, train_initialize, finetune, load_evaluate

with open(sys.argv[1], 'r') as f:
    config = load(f, Loader=FullLoader)

start_seed = config['start_seed']
end_seed = config['start_seed'] + config['n_seeds']

def mean_positive(alpha, preds):
    return preds[alpha > 0].mean()

def mean_negative(alpha, preds):
    return preds[alpha <= 0].mean()

def positive_mse(alpha, preds):
    return mean_squared_error(alpha[alpha > 0], preds[alpha > 0])

def negative_mse(alpha, preds):
    return mean_squared_error(alpha[alpha <= 0], preds[alpha <= 0])

metrics = [
    ("accuracy", accuracy_score),
    ("f1", f1_score),
    ("auc", roc_auc_score),
    ("precision", precision_score),
    ("recall", recall_score),
    ("r2", r2_score),
    ("mse", mean_squared_error),
    ("mae", mean_absolute_error),
    ("mean_positive_proportion", mean_positive),
    ("mean_negative_proportion", mean_negative),
    ("positive_mse", positive_mse),
    ("negative_mse", negative_mse)
]

print(config['param_grid'])
param_grid = ParameterGrid(config['param_grid'])
sample = param_grid[0]

os.makedirs("experiment/%s/" % config['experiment_name'], exist_ok=True)
with open('experiment/%s/results.csv' % config['experiment_name'], 'w') as f:
    writer = csv.writer(f)
    row_names = ["seed", "hash"]
    writer.writerow(row_names + list(sample.keys()) + [name + "_train" for name, _ in metrics] + [name + "_valid" for name, _ in metrics])

param_grid = ParameterGrid(config['param_grid'])
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for seed in range(start_seed, end_seed):
    dataset_train = PointsetDataset(config['datasets'], train=True, fold=0, random_state=seed, **config['dataset_config'])
    dataset_valid = PointsetDataset(config['datasets'], train=False, fold=0, random_state=seed, **config['dataset_config'])

    diff, allpos_sample, allneg_sample = dd_initialize(dataset_train, dataset_valid, **config['experiment_config'])
    for params in tqdm(param_grid):
        paramsjson = json.dumps(params).encode('utf-8')
        param_hash = hashlib.md5(paramsjson).hexdigest()

        resdir = "experiment/%s/models/seed_%d/hash_%s" % (config['experiment_name'], seed, param_hash)

        os.makedirs(resdir, exist_ok=True)

        alpha_valid = dataset_valid.proportion
        if True:
            kwargs = params.copy()
            for key in config['experiment_config']:
                kwargs[key] = config['experiment_config'][key]

            model, Xinit_train, yinit_train = train_initialize(dataset_train, dataset_valid, diff, allpos_sample, allneg_sample, **kwargs)

            torch.save((Xinit_train, yinit_train), "%s/dd_init.pt" % resdir)
            torch.save(model.cpu(), "%s/model_init.pt" % resdir)

            model, (preds_val, y_valid), (preds_train, y_train, alpha_train) = finetune(dataset_train, dataset_valid, model, **kwargs)

            torch.save(model, "%s/model.pt" % resdir)
        else:
            model, (preds_val, y_valid), (preds_train, y_train, alpha_train) = load_evaluate(dataset_train, dataset_valid, "%s/model.pt")

        train_metrics = []
        for name, metric in metrics:
            if name in ["accuracy", "f1", "precision", "recall"]:
                train_metrics.append(metric(y_train.numpy(), (preds_train > 0.02).to(int).numpy()))
            elif name in ["auc"]:
                train_metrics.append(metric(y_train.numpy(), preds_train.numpy()))
            else:
                train_metrics.append(metric(alpha_train.numpy(), preds_train.numpy()))

        val_metrics = []
        for name, metric in metrics:
            if name in ["accuracy", "f1", "precision", "recall"]:
                val_metrics.append(metric(y_valid.numpy(), (preds_val > 0.02).to(int).numpy()))
            elif name in ["auc"]:
                val_metrics.append(metric(y_valid.numpy(), preds_val.numpy()))
            else:
                val_metrics.append(metric(alpha_valid.numpy(), preds_val.numpy()))


        with open('experiment/%s/results.csv' % config['experiment_name'], 'a') as f:
            writer = csv.writer(f)
            metadata = [seed, param_hash]
            param_list = [str(v) for v in params.values()]
            writer.writerow(metadata + param_list + train_metrics + val_metrics)
