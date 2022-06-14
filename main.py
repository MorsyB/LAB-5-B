from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def read_data(filename):
    file = pandas.read_csv("data\\" + filename, header=None)
    file = file.values
    data, label = file[:, 1:-1], file[:, -1]

    label = LabelEncoder().fit_transform(label)
    print(f'data: {data}')
    print(f'label: {label}')
    return data, label


if __name__ == '__main__':
    data, label = read_data("glass.data")
    normalized = DataFrame(MinMaxScaler().fit_transform(data))

    train, test, train_vec, test_vec = train_test_split(data, label, test_size=0.2, random_state=42)
