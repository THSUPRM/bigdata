import csv
import math
import itertools
import time

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from ths.nn.metrics.multiple import *
from keras.callbacks import History
from keras.optimizers import rmsprop, adam, sgd, adagrad
from datetime import datetime
from operator import itemgetter

max_words = 10000


def getmatrixhyperparam(i=1):
    a = [
        ['learningRate', 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
        ['epochs', 5, 10, 20, 40, 60],
        ['optimizer', 'rmsprop', 'adam', 'sgd', 'adagrad']
        ]
    a = [
        ['learningRate', 0.001],
        ['epochs', 5],
        ['optimizer', 'rmsprop', 'adam', 'sgd', 'adagrad']
        ]

    b = list()

    for n in range(0, i):
        b.append(a[n][1:len(a[n])])

    b = itertools.product(*b)

    return b


def main():
    start_time_total = time.time()
    start_time_comp = datetime.now()
    np.random.seed(11)
    X_all = []
    Y_all = []
    with open("data/cleantextlabels6.csv", "r") as f:
        i = 0
        csv_file = csv.reader(f, delimiter=',')
        for r in csv_file:
            if i != 0:
                tweet = r[0]
                label = r[1]
                X_all.append(str(tweet).strip())
                Y_all.append(int(label))
            i = i + 1

    print("Data Ingested")
    print("X_all[0]: ", X_all[0])

    tokenizer = Tokenizer(num_words=max_words, oov_token='unk')
    print("Fitting data")
    tokenizer.fit_on_texts(X_all)
    X_Seq_All = tokenizer.texts_to_sequences(X_all)
    print("X_Seq_All[0]", X_Seq_All[0])

    # print("Dictionary", tokenizer.word_index)

    print("Final Conversion")
    X_all = tokenizer.sequences_to_matrix(X_Seq_All, mode='binary')
    print("train_x[0]", X_all[0])
    print("TYPE ALL:", type(X_all))
    print("LONGITUD X_ALL: ", len(X_all))
    print("train_y[0]", X_all[0])

    Y_mapped = to_categorical(Y_all, num_classes=3)

    num_data = len(X_all)

    # print("TOTAL DATA: ", num_data)

    test_count = math.floor(num_data * 0.20)

    # TEST SET -> data set for test the model
    X_test = X_all[0:test_count]
    Y_test = Y_all[0:test_count]

    # CROSS VALIDATION SET -> data set for validate the model
    X_valid = X_all[(test_count + 1):(test_count * 2)]
    Y_valid = Y_all[(test_count + 1):(test_count * 2)]

    # print("VALID: " + str(X_valid.__len__()))
    # print("Len:" + str(len(X_valid)) + " ARRAY:" + str(X_valid))

    # TRAINING SET -> data set for training and cross validation
    X_train = X_all[(test_count * 2) + 1:]
    Y_train = Y_mapped[(test_count * 2) + 1:]

    class_weight_val = class_weight.compute_class_weight('balanced', np.unique(Y_all), Y_all)
    class_weight_dictionary = {0: class_weight_val[0], 1: class_weight_val[1], 2: class_weight_val[2]}

    num_params = 3
    l = list()
    params = getmatrixhyperparam(num_params)

    models = list()
    for combination in params:
        start_time_comb = time.time()
        file_name = "models/LR/model" + str(combination).replace(" ", "") + ".txt"
        log = open(file_name, "a+")
        log.write("Start time: " + str(datetime.now()))
        log.write("\nCOMBINATION: " + str(combination))

        l = [0] * num_params
        for e in range(0, num_params):
            l[e] = combination[e]

        print("Create Model")
        model = Sequential()
        model.add(Dense(3, input_dim=10000))
        model.add(Activation('softmax'))
        model.summary()

        print("Compilation")

        opt = None
        if l[2] == 'rmsprop':
            opt = rmsprop(lr=l[0])
        elif l[2] == 'adam':
            opt = adam(lr=l[0])
        elif l[2] == 'sgd':
            opt = sgd(lr=l[0])
        elif l[2] == 'adagrad':
            opt = adagrad(lr=l[0])

        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X_train, Y_train, epochs=l[1], class_weight=class_weight_dictionary)

        predicted = model.predict(X_valid)
        predicted = np.argmax(predicted, axis=1)

        acc = accuracy(Y_valid, predicted)
        c_matrix = confusion_matrix(Y_valid, predicted)

        prec_0 = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[1][0] + c_matrix[2][0])
        prec_1 = c_matrix[1][1] / (c_matrix[1][1] + c_matrix[0][1] + c_matrix[2][1])
        prec_2 = c_matrix[2][2] / (c_matrix[2][2] + c_matrix[1][2] + c_matrix[0][2])

        recall_0 = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[0][1] + c_matrix[0][2])
        recall_1 = c_matrix[1][1] / (c_matrix[1][1] + c_matrix[1][0] + c_matrix[1][2])
        recall_2 = c_matrix[2][2] / (c_matrix[2][2] + c_matrix[2][0] + c_matrix[2][1])

        f1_0 = 2 * ((prec_0 * recall_0)/(prec_0 + recall_0))
        f1_1 = 2 * ((prec_1 * recall_1)/(prec_1 + recall_1))
        f1_2 = 2 * ((prec_2 * recall_2)/(prec_2 + recall_2))

        tn_0 = c_matrix[1][1] + c_matrix[1][2] + c_matrix[2][1] + c_matrix[2][2]
        tn_1 = c_matrix[0][0] + c_matrix[0][2] + c_matrix[2][0] + c_matrix[2][2]
        tn_2 = c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1]

        spec_0 = tn_0 / (tn_0 + c_matrix[1][0] + c_matrix[2][0])
        spec_1 = tn_1 / (tn_1 + c_matrix[0][1] + c_matrix[2][1])
        spec_2 = tn_2 / (tn_2 + c_matrix[0][2] + c_matrix[1][2])

        # print("Confusion matrix : ")
        # print(c_matrix)
        # print("Model accuracy: ", acc)
        # print("Precision 0: ", prec_0)
        # print("Precision 1: ", prec_1)
        # print("Precision 2: ", prec_2)
        # print("Recall 0: ", recall_0)
        # print("Recall 1: ", recall_1)
        # print("Recall 2: ", recall_2)
        # print("F1 Score 0: ", f1_0)
        # print("F1 Score 1: ", f1_1)
        # print("F1 Score 2: ", f1_2)
        # print("Specificity 0: ", spec_0)
        # print("Specificity 1: ", spec_1)
        # print("Specificity 2: ", spec_2)

        log.write("\nConfusion matrix :" + "\n" + str(c_matrix) + "\nModel accuracy: " + str(acc) +
                  "\nPrecision 0: " + str(prec_0) + "\nPrecision 1: " + str(prec_1) + "\nPrecision 2: "
                  + str(prec_2) + "\nRecall 0: " + str(recall_0) + "\nRecall 1: " + str(recall_1) +  "\nRecall 2: "
                  + str(recall_2) + "\nF1 Score 0: " + str(f1_0) + "\nF1 Score 1: " + str(f1_1) +  "\nF1 Score 2: "
                  + str(f1_2) + "\nSpecificity 0: " + str(spec_0) + "\nSpecificity 1: " + str(spec_1) +
                  "\nSpecificity 2: " + str(spec_2))

        model = [acc,       # Accuracy
                 prec_1,    # Precision
                 recall_1,  # Recall
                 f1_1,      # f1 Score
                 spec_1,    # Specificity
                 file_name]

        if len(models) < 8:
            models.append(model)
        else:
            models.append(model)
            models = sorted(models, key=itemgetter(3, 2, 1))
            models.pop(3)

        print("Done and Cleared Keras!")
        log.write(("\nExecution time: {} minutes".format((time.time() - start_time_comb) / 60)))
        log.write(("\nFinish time: " + str(datetime.now())))
        log.close()

    file = open("models/LR/bestModels" + str(datetime.now())[:19].replace(" ", "T") + ".txt", "a+")

    for m in models:
        file.write("--------------------------------------------------------------------------------------------")
        file.write("\nacc: "        + str(m[0]))
        file.write("\nprec_1: "     + str(m[1]))
        file.write("\nrecall_1: "   + str(m[2]))
        file.write("\nf1_1: "       + str(m[3]))
        file.write("\nspec_1: "     + str(m[4]))
        file.write("\nfileName: "   + str(m[5]))
        file.write("\n--------------------------------------------------------------------------------------------")

    file.write("\nStart TOTAL execution time: " + str(start_time_comp))
    file.write("\nTOTAL execution time: {} hours".format((time.time() - start_time_total) / (60 * 60)))
    file.write("\nFinish TOTAL execution time: " + str(datetime.now()))
    file.close()


if __name__ == "__main__":
    main()
