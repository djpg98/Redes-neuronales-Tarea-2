import csv
import random

""" Conjunto de funciones básicas que debe tener un dataset """
class DatasetMixin:

    """ Añade una componente de sesgo a cada dato en el dataset. Esto se logra
        agregando una componente adicional que siempre vale 1 a todos los datos 
        en el dataset.
    """
    def add_bias_term(self):

        for i in range(len(self.features)):

            self.features[i].insert(0, 1)

    """ Devuelve la cantidad de elementos en el dataset """
    def size(self):

        return len(self.features)

    def training_data_size(self):

        return len(self.training_data)

    def test_data_size(self):

        return len(self.test_data)

    """ Devuelve el tamaño del input vector (Incluendo el término de bias si
        este ha sido agregado
    """
    def feature_vector_length(self):

        return len(self.features[0])

    """ Iterador para todos los elementos del dataset """
    def __iter__(self):

        for pair in zip(self.features, self.values):

            yield pair

    def training_data_iter(self):

        for index in self.training_data:

            yield (self.features[index], self.values[index])

    def test_data_iter(self):

        for index in self.test_data:

            yield (self.features[index], self.values[index])

    def shuffle_training_data(self):
        random.shuffle(self.training_data)

class BinaryDataset(DatasetMixin):

    def __init__(self, datafile, positive_category):

        self.features = []
        self.values = []

        with open(datafile, 'r') as csv_file:

            data_reader = csv.reader(csv_file, delimiter=",")

            #SKip header
            next(data_reader)

            for row in data_reader:

                features, value = row[:-1], row[-1:][0]

                if positive_category == value:
                    numeric_value = 1
                else:
                    numeric_value = 0

                self.features.append(list(map(float, features)))
                self.values.append(numeric_value)

            csv_file.close()

        index_list = [i for i in range(len(self.features))]
        self.training_data = random.sample(index_list, int(0.80 * len(self.features)))
        self.test_data = [index for index in index_list if index not in self.training_data]


class MultiClassDataset(DatasetMixin):

    def __init__(self, datafile, label_dictionary):

        self.features = []
        self.values = []
        self.label_dictionary = label_dictionary

        with open(datafile, 'r') as csv_file:

            data_reader = csv.reader(csv_file, delimiter=",")
            
            #SKip header
            next(data_reader)

            for row in data_reader:

                features, value = row[1:], row[0]

                self.features.append(list(map(float, features)))
                self.values.append(value)

            csv_file.close()

        index_list = [i for i in range(len(self.features))]
        self.training_data = random.sample(index_list, int(0.80 * len(self.features)))
        self.test_data = [index for index in index_list if index not in self.training_data]

    def normalize_data(self, normalizer_function):

        for i in range(len(self.features)):
            self.features[i] = list(map(normalizer_function, self.features[i]))

    def get_label_index(self, label):
        return self.label_dictionary[label]

    def get_labels(self):

        for key in self.label_dictionary.keys():
            yield key
