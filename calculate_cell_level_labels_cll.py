import numpy as np
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid
from yaml import load, FullLoader
import sys
from tqdm import tqdm
import pandas as pd
import os

from utils.dataset import PointsetDataset
from utils.trainer import dd_initialize, train_initialize, train_head, finetune, load_evaluate

with open(sys.argv[1], 'r') as f:
    config = load(f, Loader=FullLoader)

df = pd.read_csv('experiment/%s/results.csv' % config['experiment_name'])

best_seed = df['auc_train'].argmax()

seed = int(best_seed)

print(best_seed)

folders = glob("experiment/%s/models/seed_%d/*/model.pt" % (config['experiment_name'], 99))

hash_id = folders[0].split("hash_")[1].split("/")[0]

model_fn = "experiment/%s/models/seed_%d/hash_%s/model.pt" % (config['experiment_name'], seed, hash_id)

dataset_train = PointsetDataset(config['datasets_train'], random_state=seed, **config['dataset_config'])
dataset_valid = PointsetDataset(config['datasets_valid'], random_state=seed, **config['dataset_config'])
dataset_all = PointsetDataset(config['datasets_train'] + config['datasets_valid'], random_state=seed, **config['dataset_config'])

model, res_val, res_train = load_evaluate(dataset_train, dataset_valid, model_fn)

cell_level_labels_train = model.sigmoid(model.rFF(dataset_train.X.float())).detach().numpy()
predictions_train = model(dataset_train.X.float()).detach().numpy()
cells_train = dataset_train.X.numpy()
indexes_train = dataset_train.idx.numpy()
proportion_train = dataset_train.proportion.numpy()
ys_train = dataset_train.y.numpy()

cell_level_labels_valid = model.sigmoid(model.rFF(dataset_valid.X.float())).detach().numpy()
predictions_valid = model(dataset_valid.X.float()).detach().numpy()
cells_valid = dataset_valid.X.numpy()
indexes_valid = dataset_valid.idx.numpy()
ys_valid = dataset_valid.y.numpy()

# proportion_valid = dataset_valid.proportion.numpy()

cell_level_labels = np.concatenate([cell_level_labels_train, cell_level_labels_valid], axis=0)
predictions = np.concatenate([predictions_train, predictions_valid], axis=0)
cells = np.concatenate([cells_train, cells_valid], axis=0)
indexes = np.concatenate([indexes_train.astype("str"), indexes_valid[:-24].astype("str"), np.core.defchararray.add((indexes_valid[-24:]).astype("str"), "_1")], axis=0)
proportion = np.concatenate([proportion_train, -1*np.ones_like(indexes_valid)], axis=0)
ys = np.concatenate([ys_train, ys_valid], axis=0)
dataset = np.concatenate([np.ones_like(ys_train), np.zeros_like(ys_valid)], axis=0)

os.makedirs("cell_level_labels/%s" % config["experiment_name"], exist_ok=True)

headers = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W", "CD45", "CD22", "CD5", "CD19", "CD79b", "CD3", "CD81", "CD10", "CD43", "CD38", "PCell", "Cell_Label"]
for cll, c, i in zip(cell_level_labels, cells, indexes):
    cell_class = (cll > 0.5).astype(int)
    concat_c = np.concatenate((c, cll, cell_class), axis=1)
    filename = "cell_level_labels/%s/%s.csv" % (config['experiment_name'],i)
    assert concat_c.shape[1] == len(headers)
    np.savetxt(filename, concat_c, delimiter=",", header=",".join(headers), comments="")

headers = ["Index", "Class", "Proportion", "Predicted Proportion", "Training"]
m = np.concatenate((indexes[:,np.newaxis], ys[:,np.newaxis], proportion[:,np.newaxis], predictions, dataset[:,np.newaxis]), axis=1)
filename = "cell_level_labels/%s/output.csv" % config['experiment_name']
np.savetxt(filename, m, delimiter=",", header=",".join(headers), comments="", fmt="%s")