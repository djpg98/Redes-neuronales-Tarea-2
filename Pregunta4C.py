from Dataset import MultiClassDataset
from MLP import Layer

dataset_train = MultiClassDataset('mnist_train.csv', dict([(str(i), i) for i in range(10)]))
dataset_train.normalize_data(lambda x: x/255)
classifier = Layer(
    dimension=10, 
    input_dimension=784, 
    activation_function=lambda x: 1 if x >= 0 else 0
)

classifier.train_layer(dataset_train, 50, 0.1, True)


dataset_test = MultiClassDataset('mnist_test.csv', dict([(str(i), i) for i in range(10)]))
dataset_test.normalize_data(lambda x: x/255)
classifier.eval(dataset_test)