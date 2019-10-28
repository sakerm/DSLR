import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

if __name__ == '__main__':
    try:
        # Read data from file 'filename.csv'
        df = pd.read_csv('resources/dataset_train.csv')
        first = df.loc[df['Hogwarts House'] == 'Ravenclaw']
        second = df.loc[df['Hogwarts House'] == 'Slytherin']
        three = df.loc[df['Hogwarts House'] == 'Gryffindor']
        four = df.loc[df['Hogwarts House'] == 'Hufflepuff']
        features_name = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
        #for i in range(13):
        Ravenclaw_x = first['Astronomy']
        Slytherin_x = second['Astronomy']
        Gryffindor_x = three['Astronomy']
        Hufflepuff_x = four['Astronomy']
        fig = plt.figure(figsize=(5, 5))
                #for j in range(i + 1, 13):
        Ravenclaw_y = first["Defense Against the Dark Arts"]
        Slytherin_y = second["Defense Against the Dark Arts"]
        Gryffindor_y = three["Defense Against the Dark Arts"]
        Hufflepuff_y = four["Defense Against the Dark Arts"]
        plt.xlabel("Astronomy")
        plt.ylabel("Defense Against the Dark Arts")
        plt.scatter(Ravenclaw_x, Ravenclaw_y, label = 'Ravenclaw', marker='*')
        plt.scatter(Slytherin_x, Slytherin_y, label = 'Slytherin', marker='x')
        plt.scatter(Gryffindor_x, Gryffindor_y, label = 'Gryffindor', marker='o')
        plt.scatter(Hufflepuff_x, Hufflepuff_y, label = 'Hufflepuff', marker='+')
        #plt.legend()
        fig.tight_layout()
        plt.show()
    except:
        print("Usage: python3 scatter_plot.py filename_train.csv")
        exit (-1)