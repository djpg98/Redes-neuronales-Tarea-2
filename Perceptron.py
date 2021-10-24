import random
import csv

"""La clase Perceptron representa un único perceptron"""
class Perceptron:

    """ Constructor de la clase perceptron. Inicializa aleatoriamente los pesos
        de cada input entre -0.05 y 0.05
        Parámetros:
            - input_dimension: Cantidad de inputs que recibe el perceptron
            - activation_function: Función de un parámetro que será usada como función de activación
    """
    def __init__(self, input_dimension, activation_function):

        self.weights = [random.uniform(-0.05, 0.05) for i in range(input_dimension + 1)]
        self.activation_function = activation_function

    """ Suma pesada de los inputs:
        Parámetros:
            - inputs: Vector que actúa como input del perceptron
    """
    def sum_inputs(self, inputs):

        return sum(map(lambda pair: pair[0] * pair[1], zip(self.weights, inputs)))

    """ Aplica la función de activación a la suma pesada de los inputs
        Parámetros:
            - inputs: Vector que actúa como input del perceptron
    """
    def output(self, inputs):

        return self.activation_function(self.sum_inputs(inputs))

    """ Permite ajustar los pesos del perceptron cuando hay un dato mal clasificado 
        Parámetros:
            - expected_value: Valor que se esperaba devolviera el perceptron para el dato dado
            - output_value: Valor devuelto por el perceptron para el dato dado
            - learning_rate: Tasa de aprendizaje a aplicar
            - features: El dato a partir del cual se obtuvo el resultado de output_value
    """
    def adjust_weights(self, expected_value, output_value, learning_rate, features):

        factor = learning_rate * (expected_value - output_value)

        delta = map(lambda x: factor * x, features)

        self.weights = list(map(lambda pair: pair[0] + pair[1], zip(self.weights, delta)))

    """ Salva los pesos obtenidos en el último epoch en un archivo csv
        Parámetros:
            - filename: Nombre del archivo en el que se guardaran los pesos
    """
    def save_weights(self, filename):

        with open(f'{filename}.csv', 'w') as csvfile:

            writer = csv.writer(csvfile, delimiter=",")

            writer.writerow(self.weights)

            csvfile.close()


    """ Entrena un perceptron para clasificar los datos en un dataset
        Parámetros:
            - dataset: Una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_weights: Nombre del archivo donde se guardaran los pesos. Si el nombre es el 
              string vacío no se salvan los pesos
    """
    def train_perceptron(self, dataset, epochs, learning_rate, verbose=False, save_weights=""):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.weights))

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
                if verbose:
                    print(f'Todos los datos han sido correctamente clasificados en el epoch {current_epoch + 1}')
                break
            else:
                if verbose:
                    print(f'Se ha terminado la epoch {current_epoch + 1} con {error_number} errores en {dataset.size()} muestras')

        if save_weights != "":
            self.save_weights()
        else:
            if verbose:
                print("Pesos finales: ")
                print(self.weights)



