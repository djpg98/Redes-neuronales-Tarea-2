def precision(true_positives, false_positives):

    return (true_positives/(true_positives+false_positives))

def accuracy(total_data, errors):

    return ((total_data - errors)/total_data)