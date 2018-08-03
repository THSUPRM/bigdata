from keras.preprocessing.text import Tokenizer
from ths.nn.metrics import multiple as met
from sklearn import svm

import numpy as np
import csv
import math
max_words = 10000


def main():

    X_all = []
    Y_all = []
    with open("data/cleantextlabels5.csv", "r", encoding="ISO-8859-1") as f:
        i = 0
        csv_file = csv.reader(f, delimiter=',')
        for r in csv_file:
            if i != 0:
                tweet = r[0]
                label = r[1]
                X_all.append(str(tweet).strip())
                Y_all.append(int(label))
            i = i + 1

    # print("i: ", i)
    # print("X_ALL: ", X_all[0:3])
    # print("Y_ALL: ", Y_all[0:3])

    tokenizer = Tokenizer(num_words=max_words, oov_token='unk')

    tokenizer.fit_on_texts(X_all)
    # dictionary = tokenizer.word_index
    # print("dictionary: ", dictionary)

    allWordIndices = tokenizer.texts_to_sequences(X_all)

    allWordIndices = np.asarray(allWordIndices)
    # print("allWord 1: ", allWordIndices[0:3])
    X_all = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    Y_all = np.array(Y_all)

    # print("train_X", X_all[0:5])
    # print("train_Y", Y_all[0:5])

    # print("type x: ", type(X_all))
    # print("type y: ", type(Y_all))

    num_data = len(X_all)

    # print("TOTAL DATA: ", num_data)

    test_count = math.floor(num_data * 0.20)

    # TEST SET -> data set for test the model
    X_test = X_all[0:test_count]
    Y_test = Y_all[0:test_count]

    # print("TEST: " + str(X_test))
    # print("TEST: " + str(X_test.__len__()))

    # CROSS VALIDATION SET -> data set for validate the model
    X_valid = X_all[(test_count + 1):(test_count * 2)]
    Y_valid = Y_all[(test_count + 1):(test_count * 2)]

    # X_valid = X_all[3001:4000]
    # Y_valid = Y_all[3001:4000]

    # print("VALID: " + str(X_valid.__len__()))
    # print("Len:" + str(len(X_valid)) + " ARRAY:" + str(X_valid))

    # TRAINING SET -> data set for training and cross validation
    X_train = X_all[(test_count * 2) + 1:]
    Y_train = Y_all[(test_count * 2) + 1:]

    # X_train = X_all[0:3000]
    # Y_train = Y_all[0:3000]

    # print("LEN TRAIN: " + str(X_train.__len__()))
    # print("TYPE X TRAIN: " + str(type(X_train)))
    # print("TYPE Y TRAIN: " + str(type(Y_train)))
    #
    # print("SHAPE X TRAIN: " + str(X_train.shape[0:4]))
    # print("SHAPE Y TRAIN: " + str(Y_train.shape[0:4]))
    #
    # print("Content X TRAIN: " + str(X_train[0]))
    # print("Content Y TRAIN: " + str(Y_train[0]))

    # print("LEN X TRAIN: " + str(X_train.__len__()))
    # print("LEN X VALID: " + str(X_valid.__len__()))
    # print("LEN Y VALID: " + str(Y_valid.__len__()))

    #Data partitioned in three sets train, validation and test

    #///////////////////////////////////////////////SVM///////////////////////////////////////////////#
    # print("////////////////////////////////////Support Vector Machine////////////////////////////////////")
    # clf = svm.SVC()
    # clf.fit(X_train, Y_train)
    # print("........................TRAINED SVM........................")
    #
    # # print("PREDICT: ", clf.predict(pred_X))
    #
    # predicted = clf.predict(X_valid)
    # # print("SVM Predicted: ", predicted)
    # print("........................PREDICTED SVM........................")
    #
    # for x, y in zip(Y_valid, predicted):
    #     print("SVM expected value: " + str(x) + " predicted value: " + str(y))
    #
    # from sklearn.metrics import accuracy_score
    # acc_scikit = accuracy_score(Y_valid, predicted)
    #
    # accuracy = met.accuracy(Y_valid, predicted)
    # precision = met.precision_numpy(Y_valid, predicted)
    # recall = met.recall_numpy(Y_valid, predicted)
    # f1score = met.f1_numpy(Y_valid, predicted)
    # fprate = met.fprate_numpy(Y_valid, predicted)
    #
    # print("SVM acc_scikit : ", acc_scikit)
    # print("SVM ACC : ", accuracy)
    # print("SVM PRECISION : ", precision)
    # print("SVM RECALL : ", recall)
    # print("SVM F1SCORE : ", f1score)
    # print("SVM FPRATE :", fprate)
    # print("////////////////////////////////////Support Vector Machine////////////////////////////////////")
    #///////////////////////////////////////////////SVM///////////////////////////////////////////////#

    # ///////////////////////////////////////////////NB///////////////////////////////////////////////#
    # print("////////////////////////////////////Gaussian Naive Base////////////////////////////////////")
    # from sklearn.naive_bayes import GaussianNB
    # gnb = GaussianNB()
    # gnb.fit(X_train, Y_train)
    #
    # predicted = gnb.predict(X_valid)
    # print("NB Predicted: ", predicted)
    #
    # for x, y in zip(Y_valid, predicted):
    #     print("NB expected value: " + str(x) + " predicted value: " + str(y))
    #
    # from sklearn.metrics import accuracy_score
    # acc_scikit = accuracy_score(Y_valid, predicted)
    #
    # accuracy = met.accuracy(Y_valid, predicted)
    # precision = met.precision_numpy(Y_valid, predicted)
    # recall = met.recall_numpy(Y_valid, predicted)
    # f1score = met.f1_numpy(Y_valid, predicted)
    # fprate = met.fprate_numpy(Y_valid, predicted)
    #
    # print("NB acc_scikit : ", acc_scikit)
    # print("NB ACC : ", accuracy)
    # print("NB PRECISION : ", precision)
    # print("NB RECALL : ", recall)
    # print("NB F1SCORE : ", f1score)
    # print("NB FPRATE :", fprate)
    # print("////////////////////////////////////Gaussian Naive Base////////////////////////////////////")
    #////////////////////////////////////////////////NB///////////////////////////////////////////////#

    #//////////////////////////////////////////Logistic Regression//////////////////////////////////////////#
    print("////////////////////////////////Logistic Regression////////////////////////////////")
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)

    # score = log_reg.score(X_valid, Y_valid)
    # print("SCORE", score)

    predicted = log_reg.predict(X_valid)

    for x, y in zip(Y_valid, predicted):
        print("Log_Reg expected value: " + str(x) + " predicted value: " + str(y))

    from sklearn.metrics import accuracy_score
    acc_scikit = accuracy_score(Y_valid, predicted)

    accuracy = met.accuracy(Y_valid, predicted)
    precision = met.precision_numpy(Y_valid, predicted)
    recall = met.recall_numpy(Y_valid, predicted)
    f1score = met.f1_numpy(Y_valid, predicted)
    fprate = met.fprate_numpy(Y_valid, predicted)

    print("Log_Reg acc_scikit : ", acc_scikit)
    print("Log_Reg ACC : ", accuracy)
    print("Log_Reg PRECISION : ", precision)
    print("Log_Reg RECALL : ", recall)
    print("Log_Reg F1SCORE : ", f1score)
    print("Log_Reg FPRATE :", fprate)
    print("////////////////////////////////////Logistic Regression////////////////////////////////////")
    #//////////////////////////////////////////Logistic Regression//////////////////////////////////////////#


if __name__ == "__main__":
    main()
