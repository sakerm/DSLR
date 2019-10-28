import numpy as np
import sys
import pandas as pd
from sklearn.metrics import accuracy_score

def predict_function(X, theta):
    return np.dot(X, theta.T)

def standardize(vector, mean, std):
    return (vector - mean) / std

if __name__ == '__main__':
    np.seterr(all='ignore')
    try:
        "Read file to predict"
        df = pd.read_csv(sys.argv[1])
        df = df.fillna(0)
        real_houses = df['Hogwarts House'].tolist()
        df.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Astronomy', 'Transfiguration', 'Care of Magical Creatures', 'Potions', 'Hogwarts House'], axis=1, inplace=True)
        "Load weights"
        weights = pd.read_csv('Weights.csv')
        mean = weights['Mean'].dropna()
        std = weights['Std'].dropna()
        weights.drop(['Mean', 'Std'], axis=1, inplace=True)
        features_name = ["Arithmancy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Charms","Flying"]
        for i in range(9):
            df[features_name[i]] = standardize(df[features_name[i]], mean[i], std[i])
        X = np.hstack((np.matrix(np.ones(df.shape[0])).T, df))
        p1 = predict_function(X, np.matrix(weights['Ravenclaw']))
        p2 = predict_function(X, np.matrix(weights['Slytherin']))
        p3 = predict_function(X, np.matrix(weights['Gryffindor']))
        p4 = predict_function(X, np.matrix(weights['Hufflepuff']))
        houses = []
        for i in range(df.shape[0]):
            m =  max(p1[i][0], p2[i][0], p3[i][0], p4[i][0])
            if m == p1[i][0]:
                houses.append('Ravenclaw')
            elif m == p2[i][0]:
                houses.append('Slytherin')
            elif m == p3[i][0]:
                houses.append('Gryffindor')
            else:
                houses.append('Hufflepuff')
        houses = pd.DataFrame(houses, columns= ['Hogwarts House'])
        index = pd.DataFrame([i for i in range(df.shape[0])], columns = ['Index'])
        index = index.join(houses)
        index.to_csv('houses.csv', header=True, index=False)
        if  np.nonzero(real_houses)[0].size != 0:
            print(accuracy_score(real_houses, houses))
    except:
       print("Usage: python3 logreg_predict.py filename.csv weights_file.npy")
       exit (-1)