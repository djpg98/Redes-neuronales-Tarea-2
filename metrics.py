def precision(true_positives, false_positives):

    try:
        return (true_positives/(true_positives+false_positives))
    except:
        return -1

def accuracy(total_data, errors):

    return ((total_data - errors)/total_data)