from Dataset import BinaryDataset
from Perceptron import Perceptron

dataset = BinaryDataset('iris - iris.csv', 'Iris-virginica')
classifier = Perceptron(4, lambda x: 1 if x >= 0 else 0)

classifier.train_perceptron(dataset, 100, 0.01, verbose=True)