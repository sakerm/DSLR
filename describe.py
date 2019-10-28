import pandas as pd
import numpy as np
import math
import sys

def get_column_data(vector, name):
    c, s, max_column, min_column, res = 0, 0, vector[0], vector[0], []
    for x in vector:
        if not np.isnan(x) or name == "Index":
            c += 1
            s += x
            res.append(x)
            if max_column < x:
                max_column = x
            if min_column > x:
                min_column = x
    return c, s, max_column, min_column, res

def get_std(vector, mean, count, name):
    std = 0
    for v in vector:
        if not np.isnan(v) or name == "Index":
            std += (v - mean) ** 2
    std /= (count - 1)
    return (math.sqrt(std))

def get_quartile(sorted_array, q):
    if q - int(q) == 0.25:
        return (sorted_array[math.ceil(q)] + 3 * sorted_array[math.floor(q)]) / 4
    if q - int(q) == 0.5:
        return (sorted_array[math.ceil(q)] + sorted_array[math.floor(q)]) / 2
    return (sorted_array[math.ceil(q)] * 3 + sorted_array[math.floor(q)]) / 4

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    try:
        data = pd.read_csv(sys.argv[1])
        features = {}
        features_name = ["Index", "Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
        for i in range(14):
            feature = {}
            feature["Count"], sum_column, max_column, min_column, vector = get_column_data(data[features_name[i]], features_name[i])
            feature["Mean"] = np.float32(sum_column / feature["Count"])
            feature["Std"] = get_std(data[features_name[i]], feature["Mean"], feature["Count"], features_name[i])
            feature["Min"] = min_column
            sorted_array = sorted(vector)
            q = (feature["Count"] - 1) * 0.25
            if int(q) == q:
                feature["25%"] = sorted_array[int(q)]
            else:
                feature["25%"] = get_quartile(sorted_array, q)
            q = (feature["Count"] - 1) * 0.5
            if int(q) == q:
                feature["50%"] = sorted_array[int(q)]
            else:
                feature["50%"] = get_quartile(sorted_array, q)
            q = (feature["Count"] - 1) * 0.75
            if int(q) == q:
                feature["75%"] = sorted_array[int(q)]
            else:
                feature["75%"] = get_quartile(sorted_array, q)
            feature["Max"] = max_column
            feature["Range"] = max_column - min_column
            feature["Interquartile Range"] = feature["75%"] - feature["25%"]
            features[features_name[i]] = feature
        print(pd.DataFrame.from_dict(features))
    except:
        print("Usage: python3 describe filename_train.csv")
        exit (-1)
