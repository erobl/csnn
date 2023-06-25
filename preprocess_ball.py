import numpy as np
import pandas as pd
import torch
from pathlib import Path
import os

BASE_DIR = Path("data/raw/")

RESULT_DIR = Path("data/B-ALL")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR / "train", exist_ok=True)
os.makedirs(RESULT_DIR / "test", exist_ok=True)

train_set = pd.read_csv("metadata/B-ALL_train.csv")
test_set = pd.read_csv("metadata/B-ALL_test.csv")

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
        "idx": row['Index'],
        "proportion": proportion,
        "tags": ["B-ALL", "train"],
        "feature_names": feature_names
    }

    path = RESULT_DIR / "train" / ("%d.pt" % row["Index"])

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
        "idx": row['Index'],
        "proportion": proportion,
        "tags": ["B-ALL", "test"],
        "feature_names": feature_names
    }

    path = RESULT_DIR / "test" / ("%d.pt" % row["Index"])

    torch.save(data, path)