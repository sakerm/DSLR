import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import sys

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    try:
    # Read data from file 'filename.csv'
        df = pd.read_csv(sys.argv[1])
        first = df[df['Hogwarts House'] == 'Ravenclaw']
        second = df[df['Hogwarts House'] == 'Slytherin']
        three = df[df['Hogwarts House'] == 'Gryffindor']
        four = df[df['Hogwarts House'] == 'Hufflepuff']
        sns.set(color_codes=True)
        features_name = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(20,12))
        fig.suptitle('Distribution of Students Grades for Each House', fontsize=20)
        #plt.xticks(gas.Year[::3])
        for i in range(13):
                plt.subplot(4,4,i + 2)
                Ravenclaw = first[features_name[i]]
                Slytherin = second[features_name[i]]
                Gryffindor = three[features_name[i]]
                Hufflepuff = four[features_name[i]]
                plt.hist(Ravenclaw, alpha=0.5,label = 'Ravenclaw')
                plt.hist(Slytherin, alpha=0.5,label = 'Slytherin')
                plt.hist(Gryffindor, alpha=0.5,label = 'Gryffindor')
                plt.hist(Hufflepuff, alpha=0.5,label = 'Hufflepuff')
                plt.title(features_name[i])
                plt.xlabel('Grades')
                plt.ylabel('Frequency of students')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.legend(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'],loc=2,fontsize=16,shadow=True,bbox_to_anchor=(0.05, 0.9))
        # plt.savefig('Histogram.png', dpi=300)
        plt.show()
    except:
        print("Usage: python3 histogram.py filename_train.csv")
        exit (-1)
