import numpy as np
import pandas as pd
import torch
from pathlib import Path
import os

BASE_DIR = Path("data/raw/")

RESULT_DIR = Path("data/CLL")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR / "train", exist_ok=True)
os.makedirs(RESULT_DIR / "val", exist_ok=True)
os.makedirs(RESULT_DIR / "test", exist_ok=True)
os.makedirs(RESULT_DIR / "test_noprop", exist_ok=True)

train_set = pd.read_csv("metadata/CLL_train.csv")
val_set = pd.read_csv("metadata/CLL_val.csv")
test_set = pd.read_csv("metadata/CLL_test.csv")
test_noprop_set = pd.read_csv("metadata/CLL_test_noprop.csv")

for _, row in train_set.iterrows():
    print(row['Index'])
    cells = pd.read_csv(BASE_DIR / row['File'], sep="\t")

    feature_names = list(cells.columns)
    X = torch.tensor(cells.to_numpy())
    y = int(row['Proportion'] > 0)
    proportion = row['Proportion']

    data = {
        "X": X,
        "y": y,
        "proportion": proportion,
        "tags": ["B-ALL", "train"],
        "feature_names": feature_names
    }

    path = RESULT_DIR / "train" / ("%d.pt" % row["Index"])

    torch.save(data, path)

for _, row in val_set.iterrows():
    print(row['Index'])
    cells = pd.read_csv(BASE_DIR / row['File'], sep="\t")

    feature_names = list(cells.columns)
    X = torch.tensor(cells.to_numpy())
    y = int(row['Proportion'] > 0)
    proportion = row['Proportion']

    data = {
        "X": X,
        "y": y,
        "proportion": proportion,
        "tags": ["B-ALL", "val"],
        "feature_names": feature_names
    }

    path = RESULT_DIR / "val" / ("%d.pt" % row["Index"])

    torch.save(data, path)

for _, row in test_set.iterrows():
    print(row['Index'])
    cells = pd.read_csv(BASE_DIR / row['File'], sep="\t")

    feature_names = list(cells.columns)
    X = torch.tensor(cells.to_numpy())
    y = int(row['Proportion'] > 0)
    proportion = row['Proportion']

    data = {
        "X": X,
        "y": y,
        "proportion": proportion,
        "tags": ["B-ALL", "test"],
        "feature_names": feature_names
    }

    path = RESULT_DIR / "test" / ("%d.pt" % row["Index"])

    torch.save(data, path)

for _, row in test_noprop_set.iterrows():
    print(row['Index'])
    cells = pd.read_csv(BASE_DIR / row['File'], sep="\t")

    feature_names = list(cells.columns)
    X = torch.tensor(cells.to_numpy())
    y = row['Label']

    data = {
        "X": X,
        "y": y,
        "tags": ["B-ALL", "test", "noprop"],
        "feature_names": feature_names
    }

    path = RESULT_DIR / "test_noprop" / ("%d.pt" % row["Index"])

    torch.save(data, path)