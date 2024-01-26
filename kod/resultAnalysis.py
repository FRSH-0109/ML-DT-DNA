import numpy as np

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

def calculate_positive_rates(confusion_matrix):
    true_negative = confusion_matrix[0][0]
    false_negative = confusion_matrix[0][1]
    false_positive = confusion_matrix[1][0]
    true_positive = confusion_matrix[1][1]

    true_positive_rate = true_positive / (true_positive + false_negative)
    false_positive_rate = false_positive / (false_positive + true_negative)
    return true_positive_rate, false_positive_rate

def calculate_precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)

def calculate_sensitivity(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)

def calculate_f1_score(true_positive, false_positive, false_negative):
    precision = calculate_precision(true_positive, false_positive)
    sensitivity = calculate_sensitivity(true_positive, false_negative)
    return pow((((1/precision) + (1/sensitivity)) / 2), -1)
