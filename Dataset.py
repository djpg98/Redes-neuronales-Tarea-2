import csv

class Dataset:

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
                    numeric_value = -1

                self.features.append(list(map(float, features)))
                self.values.append(numeric_value)

            csv_file.close()

    def __iter__(self):

        for pair in zip(self.features, self.values):

            yield pair