import pandas as pd
import numpy as np
import math
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def numeric_features(dataset):
    numeric_col = []
    for col_name in dataset.columns:
        try:
            float(dataset[col_name][0])
            numeric_col.append(col_name)
        except ValueError:
            continue
    return numeric_col

def pairplot(data, cols):
    cols.remove("Index")
    cols = ["Hogwarts House"] + cols
    data = data[cols]
    data = data.dropna()
    sns.pairplot(data, hue="Hogwarts House", markers = ".", height=2)
    plt.show()
    return data

if __name__ == "__main__":
    try:
        data =  pd.read_csv(sys.argv[1])
        result = numeric_features(data)
        la = pairplot(data, result)
    except:
        print("Usage: python3 pair_plot.py filename_train.csv")
        exit (-1)