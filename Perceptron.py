from os import error
import random

class Perceptron:

    def __init__(self, input_dimension, activation_function):

        self.weights = [random.uniform(-1.5, 1.5) for i in range(input_dimension)]
        self.activation_function = activation_function

    def sum_inputs(self, inputs):

        return sum(map(lambda pair: pair[0] * pair[1], zip(self.weights, inputs)))

    def output(self, inputs):

        return self.activation_function(self.sum_inputs(inputs))

    def adjust_weights(self, expected_value, output_value, learning_rate, features):

        factor = learning_rate * (expected_value - output_value)

        delta = map(lambda x: factor * x, features)

        self.weights = list(map(lambda pair: pair[0] + pair[1], zip(self.weights, delta)))

    def train_perceptron(self, dataset, epochs, learning_rate):

        for current_epoch in range(epochs):

            epoch_errors = False
            error_number = 0

            for features, expected_value in dataset:

                output_value = self.output(features)
                
                if output_value != expected_value:

                    epoch_errors = True
                    error_number += 1

                    self.adjust_weights(expected_value, output_value, learning_rate, features)


            if not epoch_errors:
                print(f'Todos los datos han sido correctamente clasificados en el epoch {current_epoch + 1}')
                print(self.weights)
                break
            else:
                print(f'Se ha terminado la epoch {current_epoch + 1} con {error_number} en {len(dataset.features)}')



