import csv
import random

"""Conjunto de funciones básicas que debe tener un dataset"""
class DatasetMixin:

    def add_bias_term(self):

        for i in range(len(self.features)):

            self.features[i].insert(0, 1)

    def size(self):

        return len(self.features)

    def feature_vector_length(self):

        return len(self.features[0])

    def __iter__(self):

        for pair in zip(self.features, self.values):

            yield pair

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

        self.iteration_order = [i for i in range(len(self.features))]
        random.shuffle(self.iteration_order)

    def shuffle_data(self):
        random.shuffle(self.iteration_order)

    def normalize_data(self, normalizer_function):

        for i in range(len(self.features)):
            self.features[i] = list(map(normalizer_function, self.features[i]))

    def get_label_index(self, label):
        return self.label_dictionary[label]

    def __iter__(self):

        for index in self.iteration_order:

            yield (self.features[index], self.values[index])