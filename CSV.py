import numpy as np
import pandas
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class CSV:
    def __init__(self,path):
        self.path=path
        self.data,self.laber=self.read_data()

    def read_data(self):
        file = pandas.read_csv("data\\" + self.path, header=None)
        file = file.values
        data, label = file[:, 1:-1], file[:, -1]

        label = LabelEncoder().fit_transform(label)
        print(f'data: {data}')
        print(f'label: {label}')
        array=[0,0,0,0,0,0]
        for i in label:
            #print (i)
            array[i]+=1
        print("Distribution is : ",array)
        #print(len(label))
        return data, label
