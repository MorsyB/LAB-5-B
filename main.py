import numpy as np
import pandas
import random

from IPython.core.display_functions import clear_output
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Relu(x):
    return x * (x > 0)





def read_data(filename):
    file = pandas.read_csv("data\\" + filename, header=None)
    file = file.values
    data, label = file[:, 1:-1], file[:, -1]

    label = LabelEncoder().fit_transform(label)
    print(f'data: {data}')
    print(f'label: {label}')
    return data, label

class generic_algorith:
    def execute(self, pop_size, generations, threshold, x, y, network):
        class Agent:
            def __init__(self, network):
                class neural_network:
                    def __init__(self, network):
                        self.weights = []
                        self.activations = []
                        for layer in network:
                            if layer[0] != None:
                                input_size = layer[0]
                            else:
                                input_size = network[network.index(layer) - 1][1]
                            output_size = layer[1]
                            activation = layer[2]
                            self.weights.append(np.random.randn(input_size, output_size))
                            self.activations.append(activation)
                    def propagate(self, data):
                        input_data = data
                        for i in range(len(self.weights)):
                            z = np.dot(input_data, self.weights[i])
                            a = self.activations[i](z)
                            input_data = a
                        yhat = a
                        return yhat
                self.neural_network = neural_network(network)
                self.fitness = 0
            def __str__(self):
                return 'LOSS: ' + str(self.fitness[0])
        def generate_agents(population, network):
            return [Agent(network) for _ in range(population)]

        def fitness(agents, x, y):
            for agentosh in agents:
                yhat = agentosh.neural_network.propagate(x)
                cost = (yhat - y) ** 2
                agentosh.fitness = sum(cost)
            return agents
        def selection(agents):
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
            print('\n'.join(map(str, agents)))
            agents = agents[:int(0.2 * len(agents))]
            return agents
        def unflatten(flattened, shapes):
            newarray = []
            index = 0
            for shape in shapes:
                size = np.product(shape)
                newarray.append(flattened[index: index + size].reshape(shape))
                index += size
            return newarray
        def crossover(agents, network, pop_size):
            offspring = []
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent(network)
                child2 = Agent(network)
                shapes = [a.shape for a in parent1.neural_network.weights]
                genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.weights])
                genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.weights])
                split = random.randint(0, len(genes1) - 1)
                child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child1.neural_network.weights = unflatten(child1_genes, shapes)
                child2.neural_network.weights = unflatten(child2_genes, shapes)
                offspring.append(child1)
                offspring.append(child2)
            agents.extend(offspring)
            return agents
        def mutation(agents):
            for agent in agents:
                if random.uniform(0.0, 1.0) <= 0.1:
                    W = agent.neural_network.weights
                    shapes = [a.shape for a in W]
                    flattened = np.concatenate([a.flatten() for a in W])
                    randint = random.randint(0, len(flattened) - 1)
                    flattened[randint] = np.random.randn()
                    newarray = []
                    indeweights = 0
                    for shape in shapes:
                        size = np.product(shape)
                        newarray.append(flattened[indeweights: indeweights + size].reshape(shape))
                        indeweights += size
                    agent.neural_network.weights = newarray
            return agents
        for i in range(generations):
            print('Generetions', str(i), ':')
            agents = generate_agents(pop_size, network)
            #print("passed generate")
            agents = fitness(agents, x, y)
            #print("passed fitnesss")
            agents = selection(agents)
            #print("i passed selection")
            agents = crossover(agents, network, pop_size)
            #print("i got passed corss over")
            agents = mutation(agents)
            #print("i got here")
            agents = fitness(agents, x, y)
            #print("but not past here")
            if any(agent.fitness < threshold for agent in agents):
                print('Threshold met at generation ' + str(i) + ' !')
            if i % 100:
                #print("hi")
                clear_output()
        return agents[0]




if __name__ == '__main__':
    data, label = read_data("glass.data")
    normalized = DataFrame(MinMaxScaler().fit_transform(data))

    train, test, train_vec, test_vec = train_test_split(normalized, label, stratify=label, test_size=0.2,
                                                        random_state=1)
    mlp = MLPClassifier(random_state=1, max_iter=8000000).fit(train, train_vec)
    print(mlp)
    predict = mlp.predict_proba(test)
    print(predict)
    i = 0
    predicted_labels = []
    for x in predict:
        index = 0
        maximal_index = 0
        softmaX = softmax(x)
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
    # with shay code:
    # x = X_train
    # y = y_train
    x = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    network = [[3, 10, Relu], [None, 1, Relu]]
   # nt = neural_network(network)
    #nt.propagate(x)
    ga = generic_algorith()
    agent = ga.execute(100, 100, 0.1, x, y, network)
    weights = agent.neural_network.weights
    #print(agent.fitness)
    #print(agent.neural_network.propagate(x))
    #print(agent.neural_network.weights)
