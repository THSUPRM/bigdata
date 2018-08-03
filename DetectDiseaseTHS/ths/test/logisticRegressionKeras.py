import csv
import numpy as np
import math

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation
from ths.nn.metrics.multiple import *
from keras.callbacks import History

max_words = 10000

def main():
        np.random.seed(11)
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
        print("train_y[0]", X_all[0])

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

        print("Create Model")
        model = Sequential()
        model.add(Dense(1, input_dim=10000))
        model.add(Activation('sigmoid'))
        model.summary()

        print("Compilation")
        model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                      metrics=['accuracy', precision, recall, f1, fprate])

        history = History()

        model.fit(X_train, Y_train, epochs=15, validation_split=0.20, callbacks=[history])

        indicators = model.evaluate(X_valid, Y_valid)

        print("METRIC NAMES: ", model.metrics_names)
        print("ACCURACY: ", indicators)

        print("Done")


if __name__ == "__main__":
    main()