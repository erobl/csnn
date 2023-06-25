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
from utils.trainer_cnn import dd_initialize, train_initialize, load_evaluate

upsample_rate = 0.05

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

predictions_train = model(dataset_train.X.float()).detach().numpy()
cells_train = dataset_train.X.numpy()
indexes_train = dataset_train.idx.numpy()
proportion_train = dataset_train.proportion.numpy()
ys_train = dataset_train.y.numpy()

predictions_valid = model(dataset_valid.X.float()).detach().numpy()
cells_valid = dataset_valid.X.numpy()
indexes_valid = dataset_valid.idx.numpy()
ys_valid = dataset_valid.y.numpy()

# proportion_valid = dataset_valid.proportion.numpy()

y_0_train = model(dataset_train.X.float())

embeddings_train = model.activation(model.rFF(dataset_train.X.float().unsqueeze(1)))

embeddings_prime_train = (embeddings_train.sum(axis=2,keepdim=True) + embeddings_train.shape[2]*upsample_rate*embeddings_train)/(embeddings_train.shape[2]*(1+upsample_rate))
embeddings_prime_train = embeddings_prime_train.transpose(1,2).squeeze(-1).flatten(0,1)

y_prime_train = model.head(embeddings_prime_train).reshape((embeddings_train.shape[0], embeddings_train.shape[2]))

delta_y_train_upsampled = (y_prime_train - y_0_train.unsqueeze(-1))

y_0_train = model(dataset_train.X.float())

embeddings_train = model.activation(model.rFF(dataset_train.X.float().unsqueeze(1)))

embeddings_prime_train = (embeddings_train.sum(axis=2,keepdim=True) - embeddings_train)/(embeddings_train.shape[2]-1)
embeddings_prime_train = embeddings_prime_train.transpose(1,2).squeeze(-1).flatten(0,1)

y_prime_train = model.head(embeddings_prime_train).reshape((embeddings_train.shape[0], embeddings_train.shape[2]))

delta_y_train = (y_prime_train - y_0_train.unsqueeze(-1))

y_0_valid = model(dataset_valid.X.float())

embeddings_valid = model.activation(model.rFF(dataset_valid.X.float().unsqueeze(1)))

embeddings_prime_valid = (embeddings_valid.sum(axis=2,keepdim=True) - embeddings_valid.shape[2]*upsample_rate*embeddings_valid)/(embeddings_valid.shape[2]*(1+upsample_rate))
embeddings_prime_valid = embeddings_prime_valid.transpose(1,2).squeeze(-1).flatten(0,1)

y_prime_valid = model.head(embeddings_prime_valid).reshape((embeddings_valid.shape[0], embeddings_valid.shape[2]))

delta_y_valid_upsampled = (y_prime_valid - y_0_valid.unsqueeze(-1))

y_0_valid = model(dataset_valid.X.float())

embeddings_valid = model.activation(model.rFF(dataset_valid.X.float().unsqueeze(1)))

embeddings_prime_valid = (embeddings_valid.sum(axis=2,keepdim=True) - embeddings_valid)/(embeddings_valid.shape[2]-1)
embeddings_prime_valid = embeddings_prime_valid.transpose(1,2).squeeze(-1).flatten(0,1)

y_prime_valid = model.head(embeddings_prime_valid).reshape((embeddings_valid.shape[0], embeddings_valid.shape[2]))

delta_y_valid = (y_prime_valid - y_0_valid.unsqueeze(-1))

delta_y = np.concatenate([delta_y_train.detach().numpy(), delta_y_valid.detach().numpy()], axis=0)
delta_y_upsampled = np.concatenate([delta_y_train_upsampled.detach().numpy(), delta_y_valid_upsampled.detach().numpy()], axis=0)
cell_labels = (delta_y < 0).astype(float)
predictions = np.concatenate([predictions_train, predictions_valid], axis=0)
cells = np.concatenate([cells_train, cells_valid], axis=0)
indexes = np.concatenate([indexes_train, indexes_valid], axis=0)
proportion = np.concatenate([proportion_train, np.ones_like(indexes_valid)], axis=0)
ys = np.concatenate([ys_train, ys_valid], axis=0)
dataset = np.concatenate([np.ones_like(ys_train), np.zeros_like(ys_valid)], axis=0)

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dy = delta_y_train_upsampled.detach().reshape((-1)).numpy()
Xt = dataset_train.X.cpu().reshape((-1,16)).numpy()
Xt.shape, dy.shape

Xtr, Xte, ytr, yte = train_test_split(Xt, dy)

best_depth=4

cls = DecisionTreeRegressor(max_depth=best_depth)
cls.fit(Xtr, ytr)

best_node = cls.tree_.value.argmax()

best_node, cls.tree_.value.max()

tree_labels = []
for cell in cells:
    dp = cls.decision_path(cell)
    tree_labels.append(dp[:,best_node].toarray().reshape(-1))

tree_labels = np.asarray(tree_labels)

os.makedirs("cell_level_labels/%s" % config["experiment_name"], exist_ok=True)

headers = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "CD45", "CD22", "CD5", "CD19", "CD79b", "CD3", "CD81", "CD10", "CD43", "CD38", "Cell_Label"]
for tree, dyu, dy, c, i in zip(tree_labels, delta_y_upsampled, delta_y, cells, indexes):
    concat_c = np.concatenate((c, dy[:,np.newaxis], dyu[:,np.newaxis], tree[:,np.newaxis]), axis=1)
    filename = "cell_level_labels/%s/%d.csv" % (config["experiment_name"], i)
    np.savetxt(filename, concat_c, delimiter=",", header=",".join(headers), comments="")


headers = ["Index", "Class", "Predicted Class", "Train"]
m = np.concatenate((indexes[:,np.newaxis], ys[:,np.newaxis], predictions[:,np.newaxis], dataset[:,np.newaxis]), axis=1)
filename = "cell_level_labels/%s/output.csv" % config["experiment_name"]
np.savetxt(filename, m, delimiter=",", header=",".join(headers), comments="")