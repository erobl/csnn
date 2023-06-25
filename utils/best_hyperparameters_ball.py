import pandas as pd
import sys


def get_best(df, cols):
    grouped = df.groupby(by=cols)
    metric = grouped.mean()["auc_valid"]
    best = metric.index[metric.argmax()]

    return best


df = pd.read_csv("../experiment/tuning_ball_cnn/results.csv")
cols = ["lr", "head_arch", "architecture"]
best_deepcellcnn = get_best(df, cols)

print("best deepcellcnn parameters:")
print("architecture: %s, head_arch: %s, lr: %s" %  best_deepcellcnn)

df = pd.read_csv("../experiment/tuning_ball_cellcnn/results.csv")
cols = ["lr", "head_arch", "dropout", "architecture"]
best_cellcnn = get_best(df, cols)
print("best cellcnn parameters:")
print("architecture: %s, dropout: %s, head_arch: %s, lr: %s" %  best_cellcnn)

df = pd.read_csv("../experiment/tuning_ball_logistic/results.csv")
cols = ["negative_penalty", "lr", "dd_threshold", "architecture"]
best_csnnclass = get_best(df, cols)
print("best csnn-class parameters:")
print("architecture: %s, dd_threshold: %s, lr: %s, negative_penalty: %s" % best_csnnclass)

df1 = pd.read_csv("../experiment/tuning_ball_reg/results.csv")
df2 = pd.read_csv("../experiment/tuning_ball_reg_seed4/results.csv")
df1 = df1[(df1["seed"] == 0) | (df1["seed"] == 1) | (df1["seed"] == 2) | (df1["seed"] == 3)]
df = pd.concat([df1, df2])
cols = ["negative_penalty", "lr", "architecture"]
best_csnnreg = get_best(df, cols)
print("best csnn-reg parameters:")
print("architecture: %s, lr: %s, negative_penalty: %s" % best_csnnreg)
