from Dataset import MultiClassDataset
from MLP import Layer

dataset = MultiClassDataset('mnist_test.csv', dict([(str(i), i) for i in range(10)]))
dataset.normalize_data(lambda x: x/255)
classifier = Layer(
    dimension=10, 
    input_dimension=784, 
    activation_function=lambda x: 1 if x >= 0 else 0
)

classifier.train_layer(dataset, 50, 0.001, True)
