from keras import backend as K
import numpy as np

EPSILON = 0.0000001


def precision(y_true, y_pred):
    # predicted positives
    predictions = K.round(y_pred)
    predicted_positives = K.sum(predictions)

    # true positives
    true_positives = K.sum(K.round(y_true * predictions))
    P = true_positives / (predicted_positives + K.epsilon())
    return P


def recall(y_true, y_pred):
    # predicted positives
    predictions = K.round(y_pred)

    # all positives
    all_positives = K.sum(y_true)

    # true positives
    true_positives = K.sum(K.round(y_true * predictions))

    R = true_positives / all_positives
    return R


def f1(y_true, y_pred):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    return 2 * ((P * R) / (P + R + K.epsilon()))


def fprate(y_true, y_pred):
    # predicted positives
    predictions = K.round(y_pred)
    predicted_positives = K.sum(predictions)

    # all positives
    all_positives = K.sum(y_true)

    # true positives
    true_positives = K.sum(K.round(y_true * predictions))

    false_positive = predicted_positives - true_positives

    # negatives
    y_false = 1 - y_true

    all_negatives = K.sum(y_false)
    fpr = false_positive / (all_negatives + K.epsilon())
    return fpr


def precision_numpy(y_true, y_pred):
    # predicted positives
    predictions = np.round(y_pred)
    predicted_positives = np.sum(predictions)

    # true positives
    true_positives = np.sum(np.round(y_true * predictions))
    P = true_positives / (predicted_positives + EPSILON)
    return P


def recall_numpy(y_true, y_pred):
    # predicted positives
    predictions = np.round(y_pred)

    # all positives
    all_positives = np.sum(y_true)

    # true positives
    true_positives = np.sum(np.round(y_true * predictions))

    R = true_positives / all_positives
    return R


def f1_numpy(y_true, y_pred):
    P = precision_numpy(y_true, y_pred)
    R = recall_numpy(y_true, y_pred)
    return 2 * ((P * R) / (P + R + EPSILON))


def fprate_numpy(y_true, y_pred):
    # predicted positives
    predictions = np.round(y_pred)
    predicted_positives = np.sum(predictions)

    # true positives
    true_positives = np.sum(np.round(y_true * predictions))

    false_positive = predicted_positives - true_positives

    # negatives
    y_false = 1 - y_true

    all_negatives = np.sum(y_false)
    fpr = false_positive / (all_negatives + EPSILON)
    return fpr


def accuracy(y_true, y_pred):
    success = (y_true == y_pred) * 1
    return sum(success)/len(y_true)
