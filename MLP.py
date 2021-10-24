from Perceptron import Perceptron

"""La clase Layer representa una capa de perceptrones. Esta muy sencilla implmentación asume
que todos los perceptrones de la capa reciben exactamente el mismo input"""
class Layer:

    """ Constructor de la clase Layer. 
        Parámetros:
            - dimension: Cantidad de perceptrones en la capa
            - input_dimension: Dimensiones del vector de input que recibirán los perceptrones de la capa
            - activation_function: Función de un solo parámetro que será usada como función de activación
              de los perceptrones
            - perceptron_list: En caso de no pasar ninguno de los parámetros anteriores, se presenta la
              opción de pasar directamente una lista de perceptrones. Nótese que si este parámetro se
              pasa junto con los demás, los demás serán ignorados
    
    """
    def __init__(self, dimension=None, input_dimension=None, activation_function=None, perceptron_list=[]):
        if perceptron_list == []:
            self.dimension = dimension
            self.neurons = [Perceptron(input_dimension, activation_function) for i in range(dimension)]
        else:
            self.dimension = len(perceptron_list)
            self.neurons = perceptron_list

    """ Aplica la función de activación a todos los perceptrones de la capa dado un dato y devuelve
        un vector (Representado con una lista) que contiene los resultados de cada perceptron
        Parámetros:
            - input_vector: Dato de entrada suministrado a la capa
    
    """
    def output(self, input_vector):

        return [perceptron.output(input_vector) for perceptron in self.neurons]

    """ Entrena una capa (Esto se utiliza en clasificadores de una sola capa, como en la pregunta 4)
        para clasificar un dataset
        Parámetros:
            - dataset: Una clase que hereda el mixin DatasetMixin (En esta tarea
              existen dos: BinaryDataset y MultiClassDataset) que carga un dataset
              de un archivo csv y permite realizar ciertas operaciones sobre el
              mismo
            - epochs: Número máximo de epochs durante el entrenamiento
            - learning_rate: Tasa de aprendizaje
            - verbose: Si se desea imprimir información de los errores en cada epoch/pesos finales
            - save_weights: Nombre del archivo donde se guardaran los pesos. Si el nombre es el 
              string vacío no se salvan los pesos (Esta parte no está funcionando porque no he
              implementado save_weights aquí)
    """
    def train_layer(self, dataset, epochs, learning_rate, verbose=False, save_weights=""):

        dataset.add_bias_term()
        assert(dataset.feature_vector_length() == len(self.neurons[0].weights))

        for current_epoch in range(epochs):

            epoch_errors = False
            error_number = 0

            for features, expected_value in dataset:

                output_value = self.output(features)

                index = dataset.get_label_index(expected_value)

                is_incorrect = False

                for i in range(len(output_value)):

                    if i == index:
                        if output_value[index] != 1:
                            is_incorrect = True
                            self.neurons[index].adjust_weights(1, 0, learning_rate, features)
                    else:
                        if output_value[i] != 0:
                            is_incorrect = True
                            self.neurons[i].adjust_weights(0, 1, learning_rate, features)

                if is_incorrect:
                    epoch_errors = True
                    error_number += 1

            if not epoch_errors:
                if verbose:
                    print(f'Todos los datos han sido correctamente clasificados en el epoch {current_epoch + 1}')
                    break
            else:
                if verbose:
                    print(f'Se ha terminado la epoch {current_epoch + 1} con {error_number} errores en {dataset.size()} muestras')
                    dataset.shuffle_data()

"""Esta clase representa un multilayer perceptron"""
class MLP:

    """ Constructor de la clase MLP
        Parámetros:
            - layer_list: Una lista de objectos de la clase Layer. Nótese que las capas deben aparecer
              en la lista en el orden que se desea se apliquen
    """
    def __init__(self, layer_list):

        self.layers = layer_list
        self.depth = len(layer_list)

    """ Genera el output del MLP. Funciona de la siguiente manera: Se itera por las capas de la red,
        en la primera iteración se recibe directamente el dato a clasificar, luego el output de esa 
        capa se utiliza como el input de la siguiente capa y así sucesivamente. Devuelve el output que
        da la última capa después de finalizar este proceso
        Parámetros:
            - input_vector: Vector que representa el dato a clasificar
    """
    def output(self, input_vector):

        next_layer_input = input_vector

        for layer in self.layers:

            next_layer_input = layer.output(next_layer_input)

        return next_layer_input

