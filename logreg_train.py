import pandas as pd
import numpy as np
import math
import sys
import argparse
import matplotlib.pyplot as plt

def logistic_func(theta, X): 
    "logistic(sigmoid) function"
    return 1.0 / (1 + np.exp(-1 * np.dot(X, theta.T))) 

def log_gradient(theta, X, y): 
    "logistic gradient function"
    first_calc = logistic_func(theta, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X) 
    return final_calc 

def cost_func(theta, X, y):
    "cost function, J"
    log_func_v = logistic_func(theta, X)
    error = np.sum(np.power((np.ravel(log_func_v) - y),2))
    step1 = y * np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v)
    # A checker
    if np.isnan(step2):
        step2 = 0
    final = -step1 - step2
    return np.mean(final), error

# Make a prediction with thetas with a single row
def logistic_func_2(row, theta):
    yhat = 0
    for i in range(len(row)):
        yhat += theta[i] * row[i]
    return 1.0 / (1.0 + np.exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def grad_stoch(X, y, lr=0.3):
    "gradient stochastic function"
    theta = [0.0 for i in range(X.shape[1])]
    for epoch in range(100):
        sum_error = 0
        for index,row in enumerate(X):
            row_list = row.tolist()[0]
            yhat = logistic_func_2(row_list, theta)
            error = y[index] - yhat
            sum_error += error.mean()**2
            for i in range(len(row_list)):
                theta[i] = theta[i] + lr * error * yhat * (1.0 - yhat) * row_list[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lr, sum_error))
    return theta

def grad_desc(X, value, lr=5e-05):
    "gradient descent function"
    theta = np.matrix(np.zeros(X.shape[1])) 
    costs, errors = [], []
    for i in range(100):
        cost, error = cost_func(theta, X, value)
        theta = theta - (lr * log_gradient(theta, X, value))
        costs.append(cost)
        errors.append(error)
    return theta.tolist()[0], costs, errors

def scale_data(df):
    return (df - df.mean()) / df.std()

def logistic_regression(houseName, df,stochastic=False):
    value = []
    for house in df['Hogwarts House']:
        if house == houseName:
            value.append(1)
        else:
            value.append(0)
    value  = np.asarray(value)
    df.drop(['Hogwarts House'], axis=1, inplace=True)
    "scale data"
    df = scale_data(df)
    "important ;)"
    X = np.hstack((np.matrix(np.ones(df.shape[0])).T, df))
    if stochastic == False:
        return grad_desc(X, value)
    return grad_stoch(X, value), [], []


if __name__ == '__main__':
    np.seterr(divide = 'ignore')
    try:
        "Argparser"
        parser = argparse.ArgumentParser(description='Logistic regression : training')
        parser.add_argument("filename")
        parser.add_argument("-sg", action='store_true', help="stochastic gradient")
        args = parser.parse_args()
        "Read parameter file"
        df = pd.read_csv(args.filename)
        "Clean data"
        df.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Astronomy', 'Transfiguration', 'Care of Magical Creatures', 'Potions'], axis=1, inplace=True)
        "Apply Multi-classification with logistic regression: one-vs-all"
        theta_dic = {}
        "Data of Ravenclaw house"
        df = df.dropna()
        theta_dic['Ravenclaw'], costs_raven, errors_raven = logistic_regression('Ravenclaw', df.copy(), args.sg)
        theta_dic['Slytherin'], costs_sly, errors_sly = logistic_regression('Slytherin', df.copy(), args.sg)
        theta_dic['Gryffindor'], costs_gryf, errors_gryf = logistic_regression('Gryffindor', df.copy(), args.sg)
        theta_dic['Hufflepuff'], costs_huff, errors_huff = logistic_regression('Hufflepuff', df.copy(), args.sg)
        theta = pd.DataFrame.from_dict(theta_dic)
        mean = pd.DataFrame(df.mean().tolist(), columns= ['Mean'])
        std = pd.DataFrame(df.std().tolist(), columns= ['Std'])
        mean = mean.join(std)
        theta = theta.join(mean)
        theta.to_csv('weights.csv', index = None, header=True)
        if len(costs_raven) > 0:
                fig, axs = plt.subplots(1, 2, figsize=(8,4))
                axs[0].set_title('Cost Function')
                axs[0].plot(costs_raven, color='red', label ='Ravenclaw')
                axs[0,].plot(costs_sly, color='blue', label='Slytherin')
                axs[0].plot(costs_gryf, color='green', label='Gryffindor')
                axs[0].plot(costs_huff, color='yellow', label='Hufflepuff')
                axs[1].set_title('Error Function')
                axs[1].plot(errors_raven, color='red', label ='Ravenclaw')
                axs[1].plot(errors_sly, color='blue', label='Slytherin')
                axs[1].plot(errors_gryf, color='green', label='Gryffindor')
                axs[1].plot(errors_huff, color='yellow', label='Hufflepuff')
                axs[1].legend()
                axs[0].legend()
                plt.show()
    except:
        print("Usage: python3 logreg_train.py resources/dataset_train.csv")
        exit (-1)