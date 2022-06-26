import random
import numpy as np
from IPython.core.display_functions import clear_output
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

RELU = 'relu'
TANH = 'tanh'


class NeuralNetwork:
    def __init__(self, depth, hidden, activate):
        self.depth = depth
        self.hidden = hidden
        self.activateFunction = activate


class Citizen:
    def __init__(self):
        hidden = []
        depth = random.randint(1, 10)
        for i in range(depth):
            tmp = random.randint(2, 200)
            hidden.append(tmp)

        if random.randint(0, 2) == 0:
            activate = RELU
        else:
            activate = TANH

        self.network = NeuralNetwork(depth, hidden, activate)
        self.fitness = -1
        self.reg = 0


class GA:
    def __init__(self, popSize, maxIter, X_train, Y_train, X_test, Y_test):
        self.population = []
        self.buffer = []
        self.popSize = popSize
        self.eliteRate = 0.1
        self.maxIter = maxIter
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.initPopulation()

    def initPopulation(self):
        for _ in range(self.popSize):
            self.population.append(Citizen())
            self.buffer.append(Citizen())

    def calcFitness(self):
        for i in range(self.popSize):
            cls = MLPClassifier(hidden_layer_sizes=self.population[i].network.hidden, max_iter=69000,
                                activation=self.population[i].network.activateFunction, solver='adam', random_state=1)
            cls.fit(self.X_train, self.Y_train)
            predict = cls.predict(self.X_test)
            cMat = confusion_matrix(predict, self.Y_test)
            sum = cMat.sum()
            dSum = cMat.trace()
            self.population[i].fitness = dSum / sum
            self.population[i].reg = 0

    def sortByFitness(self):
        self.population.sort(key=lambda x: - x.fitness)

    def mate(self):
        for i in range(self.popSize):
            i1 = random.randint(0, self.popSize - 1)
            i2 = random.randint(0, self.popSize - 1)
            while i2 == i1:
                i2 = random.randint(0, self.popSize - 1)
            spos = random.randint(0, min(self.population[i1].network.depth, self.population[i2].network.depth))
            self.buffer[i].network.hidden = self.population[i1].network.hidden[0: spos] + self.population[
                                                                                              i2].network.hidden[spos:]
            self.buffer[i].network.depth = len(self.buffer[i].network.hidden)

    def swap(self):
        temp = self.population
        self.population = self.buffer
        self.buffer = temp

    def printBest(self, best):
        print()
        print("BEST CITIZEN :")
        print("FITNESS = ", best.fitness)

    def run(self):
        self.calcFitness()
        self.sortByFitness()
        best = self.population[0]
        for i in range(self.maxIter):
            self.calcFitness()
            self.sortByFitness()
            if self.population[0].fitness > best.fitness:
                best = self.population[0]
            self.printBest(best)
            self.mate()
        print()
        # print("Best solution overall:")
        print("Best solution accuracy:")
        print(best.fitness)
        print("Train accuracy:")
        print(best.fitness)
        print('Best solution depth: ', best.network.depth)
        print('Best solution layers: ', best.network.hidden)
        print('Best solution activation: ', best.network.activateFunction)
        cls = MLPClassifier(hidden_layer_sizes=self.population[i].network.hidden, max_iter=69000,
                            activation=self.population[i].network.activateFunction, solver='adam', random_state=1)
        cls.fit(self.X_train, self.Y_train)
        predict = cls.predict(self.X_test)
        print(classification_report(self.Y_test, predict,zero_division=0))
