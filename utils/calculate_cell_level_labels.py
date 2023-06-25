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
import pandas as pd

from utils.dataset import PointsetDataset
from utils.trainer import dd_initialize, train_initialize, train_head, finetune, load_evaluate

with open(sys.argv[1], 'r') as f:
    config = load(f, Loader=FullLoader)

df = pd.read_csv('experiment/%s/results.csv' % config['experiment_name'])

print(df)

best_seed = df['auc_train'].argmax()

seed = best_seed

print(best_seed)

dataset_train = PointsetDataset(config['datasets_train'], random_state=seed, **config['dataset_config'])
dataset_valid = PointsetDataset(config['datasets_valid'], random_state=seed, **config['dataset_config'])
dataset_all = PointsetDataset(config['datasets_train'] + config['datasets_valid'], random_state=seed, **config['dataset_config'])

folders = glob("experiment/%s/models/seed_%d/*/model.pt" % config['experiment_name'], seed)

print(folders)

model, res_val, res_train = load_evaluate(dataset_train, dataset_valid, folders[0])