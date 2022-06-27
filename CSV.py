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

    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)


    def calc_acc(self,train,train_vec,test,test_vec):
        mlp = MLPClassifier(random_state=1, max_iter=8000000).fit(train, train_vec)

        print(mlp)
        predict = mlp.predict_proba(test)
        print(predict)
        i = 0
        predicted_labels = []

        for x in predict:
            index = 0
            maximal_index = 0
            softmaX = self.softmax(x)
            max = -1

            print(f'{i}:   {softmaX}')
            for j in softmaX:
                if j > max:
                    max = j
                    maximal_index = index
                index += 1

            predicted_labels.append(maximal_index)
            i += 1

        print(f'softmax prediction:   {predicted_labels}')
        ##micro
        number_of_answer = [0, 0, 0, 0, 0, 0]
        number_of_correct = [0, 0, 0, 0, 0, 0]
        TP = 0
        FP = 0
        for i, j in zip(predicted_labels, test_vec):
            # print(f'i:   {i},   j:   {j}')
            number_of_answer[i] += 1  # mnzed akmn jwab 3ena mnhad elno3
            if i == j:
                number_of_correct[i] += 1
                TP += 1
        print(f'Micro: {TP / 43}')

        # Macro
        final_grade = 0
        for i, j in zip(number_of_answer, number_of_correct):
            x = (j / i)
            final_grade += x
            # print(f'i:   {i},   j:   {j}')
        print(f'Macro: {final_grade / 6}')
