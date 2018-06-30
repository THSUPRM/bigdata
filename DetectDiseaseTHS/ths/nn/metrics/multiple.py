from keras import backend as K

############################# First Implementation #############################
# def f1(y_true, y_pred):
#     prec = precision(y_true, y_pred)
#     rec = recall(y_true, y_pred)
#     return 2*((prec * rec)/(prec + rec + K.epsilon()))
#
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def tprate(y_true, y_pred):
#     return recall(y_true, y_pred)
#
# def fprate2(y_true, y_pred):
#     #invert true and negative so negative is 1  and true is 1.
#     y_true = 1 - y_true
#     #invert predictions so that we get 1 for the predictions originally set to 0
#     y_pred = 1 - y_pred
#     return recall(y_true, y_pred)
#
# def fprate(y_true, y_pred):
#     # true positives
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     # predicted_positives = true_positives + false_positives
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     # false_positives
#     false_positive = predicted_positives - true_positives
#     # Now work on negatives
#     y_false = 1 - y_true
#     possible_negatives = K.sum(K.round(K.clip(y_false, 0, 1)))
#     fprate = false_positive / (possible_negatives  + K.epsilon())
#     return fprate

############################# Second Implementation #############################
def precision(y_true, y_pred):
    #predicted positives
    predictions = K.round(y_pred)
    predicted_positives = K.sum(predictions)

    #true positives
    true_positives = K.sum(K.round(y_true * predictions))
    P = true_positives / (predicted_positives + K.epsilon())
    return P

def recall(y_true, y_pred):
    #predicted positives
    predictions = K.round(y_pred)

    #all positives
    all_positives = K.sum(y_true)

    #true positives
    true_positives = K.sum(K.round(y_true * predictions))

    R = true_positives / all_positives
    return R

def f1(y_true, y_pred):
    P = precision(y_true, y_pred)
    R = recall(y_true, y_pred)
    return 2*((P*R)/(P+R+K.epsilon()))

def fprate(y_true, y_pred):
    #predicted positives
    predictions = K.round(y_pred)
    predicted_positives = K.sum(predictions)

    #all positives
    all_positives = K.sum(y_true)

    #true positives
    true_positives = K.sum(K.round(y_true * predictions))

    false_positive = predicted_positives - true_positives

    #negatives
    y_false = 1 - y_true

    all_negatives = K.sum(y_false)
    fpr = false_positive / (all_negatives + K.epsilon())
    return fpr