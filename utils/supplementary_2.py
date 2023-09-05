import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from umap import UMAP
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

dim_names = [
    ["FSC-A", "SSC-A"],
    ["FSC-A", "FSC-H"],
    ["SSC-H", "CD45"],
    ["CD5", "CD19"],
    ["CD10", "CD79b"],
    ["CD22", "CD38"],
    ["CD43", "CD81"],
    ["CD3", "CD19"],
    ["CD5", "CD3"],
    ["UMAP x", "UMAP y"]
]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#2271B2","#E69F00"])

def plot_sample(df, axs, dim_names, color_name="Cell_Label"):
    df = df.sample(frac=1, random_state=0)
    for (d1, d2), ax in zip(dim_names, axs):
        ax.set_xlabel(d1)
        ax.set_ylabel(d2)
        # ax.scatter(df[d1], df[d2], c=df[color_name], s=1)
        # mask = df[color_name] < 0.5
        ax.scatter(df[d1], df[d2], c=df[color_name], cmap=cmap, s=1)
        # ax.scatter(df[d1][~mask], df[d2][~mask], c="#E69F00", s=1)
        if "UMAP" not in d1:
            ax.set_xlim(0, 4096)
        if "UMAP" not in d2:
            ax.set_ylim(0, 4096)

def draw_fig4(folders, names, sample_number, cols, type_, dim_names, scaler=None, reducer=None, dims=None):
    figsize=3.5
    n_samples = len(folders)
    fig, axs = plt.subplots(nrows=n_samples, ncols=len(dim_names)+1, figsize=((1+len(dim_names))*figsize, n_samples*figsize))
    plt.suptitle("%s Sample %s" % (type_, sample_number))

    for row, folder, name, col in zip(axs, folders, names, cols):
        df = pd.read_csv("/home/roblesee/csnn/cell_level_labels/%s/%s.csv" % (folder, sample_number))#.sample(27000, random_state=42)
        if reducer is not None:
            print("transforming data to umap...")
            df_data = df[
                dims
            ].values
            scaled_df_data = scaler.transform(df_data)
            embeddings = reducer.transform(scaled_df_data)

            df["UMAP x"] = embeddings[:,0]
            df["UMAP y"] = embeddings[:,1]
            print("transformed data to umap!")
            
        plot_sample(df, row[1:], dim_names, color_name=col)
        
        row[0].axis("off")
        row[0].text(0.5, 0.5, name,
            horizontalalignment='center',
            verticalalignment='center',
            size=20,
            transform=row[0].transAxes)

    plt.tight_layout()

folders = [
    "best_cll_cellcnn",
    "best_cll_cnn",
    "best_cll_reg"
]

cols = [
    "Tree_Label",
    "Tree_Label",
    "Cell_Label"
]

names = [
    "CellCNN",
    "DeepCellCNN",
    "Our method"
]

from glob import glob

fns = [(x.split("/")[-1].split(".")[0]) for x in glob("/home/roblesee/csnn/cell_level_labels/best_cll_reg/*.csv") if "output" not in x]

def do_work(fns_sp):
    sampled_df = pd.concat([pd.read_csv("/home/roblesee/csnn/cell_level_labels/%s/%s.csv" % ("best_cll_reg", fn)).sample(500, random_state=42) for fn in fns], ignore_index=True)

    dims = [
        "FSC-A",
        "FSC-H",
        "SSC-A",
        "SSC-H",
        "CD45",
        "CD22",
        "CD5",
        "CD19",
        "CD79b",
        "CD3",
        "CD81",
        "CD10",
        "CD43",
        "CD38"
    ]

    df_data = sampled_df[
        dims
    ].values
    scaler = StandardScaler()
    scaled_df_data = scaler.fit_transform(df_data)

    reducer = UMAP(random_state=42)
    print("fitting umap...")
    reducer.fit(scaled_df_data)
    print("fit umap!")
    for n in tqdm(fns_sp):
        draw_fig4(folders, names, n, cols, "CLL", dim_names, scaler, reducer, dims)
        print("saving figure...")
        plt.savefig("../supplementary_2/CLL_%s.png" % n)
        print("saved figure!")
        plt.close()

n_jobs = 20
fns_split = np.array_split(fns, n_jobs)

Parallel(n_jobs=n_jobs)(delayed(do_work)(x) for x in fns_split)