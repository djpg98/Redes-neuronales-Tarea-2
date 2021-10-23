from Dataset import Dataset
from Perceptron import Perceptron

dataset = Dataset('iris - iris.csv', 'Iris-setosa')
classifier = Perceptron(4, lambda x: 1 if x >= 0 else -1)

classifier.train_perceptron(dataset, 100, 0.01)