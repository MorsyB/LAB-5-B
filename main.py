import pandas


def read_data(filename):
    data = open("data\\glass.data")
    data = pandas.read_csv("data\\glass.data", header=None)
    print(data.shape)


if __name__ == '__main__':
    read_data("qwe")
