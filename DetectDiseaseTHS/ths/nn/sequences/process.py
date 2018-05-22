from ths.nn.sequences.tweets import *
from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, PadSentences
from keras import backend as KerasBack
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow import gfile as gf
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History
from sklearn.model_selection import GridSearchCV
from operator import itemgetter
from datetime import datetime

import numpy as np
import csv
import math
import itertools
import json
import time

class ProcessTweetsGlove:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def process(self, json_filename, h5_filename):
        if (gf.Exists('/tmp/logs')):
            gf.DeleteRecursively('/tmp/logs')
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        # divide the data into X_train, Y_train, X_test, Y_test
        X_train_sentences = X_all[0: limit]
        Y_train = Y_all[0: limit]
        X_test_sentences = X_all[limit:]
        Y_test = Y_all[limit:]
        print("Data Divided")
        #Get embeeding
        G = GloveEmbedding(self.embedding_filename)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len  = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        print("Train data convert to numpy arrays")
        NN = TweetSentiment2LSTM2Dense(max_len, G)
        print("model created")
        NN.build(first_layer_units = 128, dense_layer_units=1, first_layer_dropout=0, second_layer_dropout=0)
        print("model built")
        NN.summary()
        sgd = SGD(lr=0.3, momentum=0.001, decay=0.01, nesterov=False)
        adam = Adam(lr=0.03)
        #NN.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer=adam)
        NN.compile(loss="binary_crossentropy", metrics=['binary_accuracy'], optimizer='rmsprop')

        print("model compiled")
        print("Begin training")
        callback = TensorBoard(log_dir="/tmp/logs")
        NN.fit(X_train, Y_train, epochs=80, callbacks=[callback])
        print("Model trained")
        X_test_indices, max_len = S.map_sentence_list(X_test_sentences)
        print("Test data mapped")
        X_test_pad = P.pad_list(X_test_indices)
        print("Test data padded")
        X_test = np.array(X_test_pad)
        Y_test = np.array(Y_test)
        print("Test data converted to numpy arrays")
        loss, acc = NN.evaluate(X_test, Y_test)
        print("accuracy: ", acc, ", loss: " , loss)
        T = "I have a bad case of vomit"
        X_Predict = ["my zika is so bad", "i love colombia", "my has been tested for ebola", "there is a diarrhea outbreak in the city"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i)+ ": ", s)
            i = i + 1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        print("Storing model and weights")
        NN.save_model(json_filename, h5_filename)
        print("Done!")

class ProcessTweetsGloveOnePassHyperParam:
    def __init__(self, labeled_tweets_filename, embedding_filename, optimizer):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename
        self.optimizer = optimizer


    def getmatrixhyperparam(self, i=1):
        a = [
            ['learningRate' ,0.0001, 0.0009, 0.001, 0.006, 0.01, 0.05, 0.08, 0.1, 0.4, 0.7, 1],
            ['momentum'     ,0.09, 0.0009, 0.001, 0.006, 0.01, 0.05, 0.9, 1],
            ['epochs'       ,1, 10, 30, 70, 85, 100, 120],
            ['batchSize'    ,0, 5, 32, 50, 64, 80, 100],
            #LSTM1
            ['layerUnits1'  ,10, 30, 48, 60, 80, 100],
            ['kernelReg1'   ,0, 0.0001, 0.0009, 0.001, 0.006, 0.01, 0.05, 0.08, 0.1, 0.4, 0.7, 1],
            ['recuDropout1' ,0, 0.0001, 0.0009, 0.001, 0.006, 0.01, 0.05, 0.08, 0.1, 0.4, 0.7, 1],
            #Dropout1
            ['dropout1'     ,0, 0.01, 0.09, 0.1, 0.4, 0.8, 1],
            #LSTM2
            ['layerUnits2'  ,10, 30, 48, 60, 80, 100],
            ['kernelReg2'   ,0, 0.0001, 0.0009, 0.001, 0.006, 0.01, 0.05, 0.08, 0.1, 0.4, 0.7, 1],
            ['recuDropout2' ,0, 0.0001, 0.0009, 0.001, 0.006, 0.01, 0.05, 0.08, 0.1, 0.4, 0.7, 1],
            #Dropout2
            ['dropout2'     ,0, 0.01, 0.09, 0.1, 0.4, 0.8, 1],
            # DenseLayer1
            ['denseLayer1'  ,1, 5, 10, 30, 40, 60],
            ['regulaDense1' ,0, 0.0001, 0.0009, 0.001, 0.006, 0.01, 0.05, 0.08, 0.1, 0.4, 0.7, 1],
            #DenseLayer2
            ['denseLayer2'  ,3],
            #Optimizer
            ['optimizer', self.optimizer]
            ]

        b = list()

        for n in range(0, i):
            b.append(a[n][1:len(a[n])])

        b = itertools.product(*b)

        return b

    def process(self, json_filename, h5_filename):
        start_time_total = time.time()
        start_time_comp = datetime.now()
        if (gf.Exists('/tmp/logs')):
            gf.DeleteRecursively('/tmp/logs')
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter=',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1

        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        #Get embeeding
        G = GloveEmbedding(self.embedding_filename)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len = S.map_sentence_list(X_train_sentences)
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)

        NN = TweetSentiment2LSTMHyper(max_len, G)

        num_params = 16
        l = list()
        params = self.getmatrixhyperparam(num_params)
        models = list()
        # for combination in itertools.islice(params, 7):
        for combination in params:
            start_time_comb = time.time()

            log = open("models/model" + str(combination).replace(" ", "") + ".txt", "a+")
            log.write("Start time: " + str(datetime.now()))
            log.write("\nCOMBINATION: " + str(combination))

            l = [0] * num_params
            for e in range(0, num_params):
                l[e] = combination[e]

            desc = NN.build(layer_units_1=l[4], kernel_reg_1=l[5], recu_dropout_1=l[6], dropout_1=l[7],
                            layer_units_2=l[8], kernel_reg_2=l[9], recu_dropout_2=l[10], dropout_2=l[11],
                            dense_layer_1=l[12], regula_dense_1=l[13], dense_layer_2=l[14])

            NN.summary()

            # Assign the parameters agree the optimizer to use
            params_compile = {}
            if l[15] == 'SGD':
                sgd = SGD(lr=l[0], momentum=l[1], nesterov=False)
                params_compile['optimizer'] = sgd
                desc = desc + "\nSGD with learning rate: " + str(l[0]) + " momentum: " + str(l[1]) + " and nesterov=False"
            elif l[15] == 'RMSPROP':
                rmsprop = RMSprop(lr=l[0], rho=0.9, epsilon=1e-06)
                params_compile['optimizer'] = rmsprop
                desc = desc + "\nRMSPROP with learning rate: " + str(l[0]) + " rho: 0.9 and epsilon=1e-06"
            elif l[15] == 'ADADELTA':
                adadelta = Adadelta(lr=l[0], rho=0.95, epsilon=1e-06)
                params_compile['optimizer'] = adadelta
                desc = desc + "\nADADELTA with learning rate: " + str(l[0]) + " rho: 0.95 and epsilon=1e-06"
            elif l[15] == 'ADAM':
                adam = Adam(lr=l[0], beta_1=0.9, beta_2=0.999)
                params_compile['optimizer'] = adam
                desc = desc + "\nADAM with learning rate: " + str(l[0]) + " beta_1=0.9, beta_2=0.999"

            NN.compile(loss="categorical_crossentropy", metrics=['accuracy'], **params_compile)

            history = History()
            desc = desc + "\nEpochs: " + str(l[2]) + " Batch Size: " + str(l[3])

            # Assign batch size to fit function
            params_fit = {}
            if l[3] != 0:
                params_fit['batch_size'] = l[3]

            NN.fit(X_train, Y_train, epochs=l[2], validation_split=0.3, callbacks=[history], **params_fit)

            model = [history.history['acc'][0], history.history['val_acc'][0],
                     history.history['acc'][0]-history.history['val_acc'][0], desc, NN.model.get_weights(), NN.model.to_json()]

            log.write("\nacc: " + str(history.history['acc'][0]) + "\nval_acc: " + str(history.history['val_acc'][0]) +
                      "\ndiff_acc: " + str(history.history['val_acc'][0]-history.history['acc'][0]) + "\nModel:" + str(desc))

            if len(models) < 6:
                models.append(model)
            else:
                models.append(model)
                models = sorted(models, key=itemgetter(1, 0, 2))
                models.pop(3)

            # print("HISTORY history: " + str(history.history['acc'][0]-history.history['val_acc'][0]))
            #
            # print("Accuracy:::::::::::::::::::::::::" + str(history.history['acc'][0]))
            # print("VAL Accuracy:::::::::::::::::::::::::" + str(history.history['val_acc'][0]))

            # X_Predict = ["my zika is so bad but i am so happy because i am at home chilling", "i love colombia but i miss the flu food",
            #              "my has been tested for ebola", "there is a diarrhea outbreak in the city", "i got flu yesterday and now start the diarrhea",
            #              "my dog has diarrhea", "do you want a flu shoot?", "diarrhea flu ebola zika", "the life is amazing",
            #              "everything in my home and my cat stink nasty", "i love you so much", "My mom has diarrhea of the mouth",
            #              "when u got bill gates making vaccines you gotta wonder why anyone is allowed to play with the ebola virus? " +
            #              "let's just say for a second that vaccines were causing autism and worse, would they tell you? would they tell you we have more disease then ever?"]
            # X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
            # i =0
            # for s in X_Predict_Idx:
            #     print(str(i) + ": ", s)
            #     i = i+1
            # print(X_Predict)
            # X_Predict_Final = P.pad_list(X_Predict_Idx)
            # #X_Predict = [X_Predict]
            # X_Predict_Final = np.array(X_Predict_Final)
            # print("Predict: ", NN.predict(X_Predict_Final))
            # print("Storing model and weights")
            # NN.save_model(json_filename, h5_filename)

            KerasBack.clear_session()
            print("Done and Cleared Keras!")
            log.write(("\nExecution time: {} minutes".format((time.time() - start_time_comb) / 60)))
            log.write(("\nFinish time: " + str(datetime.now())))
            log.close()

        file = open("models/bestModel" + str(datetime.now())[:19].replace(" ", "T") + ".txt", "a+")

        for m in models:
            file.write("----------------------------------------------------------------------------------------------")
            file.write("\nacc: " + str(m[0]))
            file.write("\nval_acc: " + str(m[1]))
            file.write("\ndiff_acc: " + str(m[2]))
            file.write("\nmodel: " + str(m[3]))
            file.write("\nweights: " + str(m[4]))
            file.write("\nJSON: " + json.dumps(m[5], indent=4, sort_keys=True))
            file.write("\n----------------------------------------------------------------------------------------------")

        file.write("\nStart TOTAL execution time: " + str(start_time_comp))
        file.write("\nTOTAL execution time: {} hours".format((time.time() - start_time_total) / (60 * 60)))
        file.write("\nFinish TOTAL execution time: " + str(datetime.now()))
        file.close()

class ProcessTweetsGloveOnePass:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def process(self, json_filename, h5_filename):
        if (gf.Exists('/tmp/logs')):
            gf.DeleteRecursively('/tmp/logs')
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        #Get embeeding
        G = GloveEmbedding(self.embedding_filename)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        NN = TweetSentiment2LSTMMaxDenseBidirectional(max_len, G)
        print("Model created")
        NN.build(layer_units_1=48, kernel_reg_1=0.001, recu_dropout_1=0.4, dropout_1=0.3, dense_layer_1=100,
                 layer_units_2=24, kernel_reg_2=0.001, recu_dropout_2=0.2, dropout_2=0.5, dense_layer_2=20)

        print("Model built")
        NN.summary()
        rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
        NN.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=['accuracy'])
        print("Model compiled")
        print("Begin training")
        history = History()
        callback = TensorBoard(log_dir="/tmp/logs")
        NN.fit(X_train, Y_train, epochs=50, validation_split=0.3, callbacks=[history])

        print("Accuracy:::::::::::::::::::::::::" + str(history.history['acc']))
        print("VAL Accuracy:::::::::::::::::::::::::" + str(history.history['val_acc']))
        print("Model trained")
        X_Predict = ["my zika is so bad but i am so happy because i am at home chilling", "i love colombia but i miss the flu food",
                     "my has been tested for ebola", "there is a diarrhea outbreak in the city", "i got flu yesterday and now start the diarrhea",
                     "my dog has diarrhea", "do you want a flu shoot?", "diarrhea flu ebola zika", "the life is amazing",
                     "everything in my home and my cat stink nasty", "i love you so much", "My mom has diarrhea of the mouth",
                     "when u got bill gates making vaccines you gotta wonder why anyone is allowed to play with the ebola virus? " +
                     "let's just say for a second that vaccines were causing autism and worse, would they tell you? would they tell you we have more disease then ever?"]
        X_Predict_Idx, max_len2 = S.map_sentence_list(X_Predict)
        i =0
        for s in X_Predict_Idx:
            print(str(i) + ": ", s)
            i = i+1
        print(X_Predict)
        X_Predict_Final = P.pad_list(X_Predict_Idx)
        #X_Predict = [X_Predict]
        X_Predict_Final = np.array(X_Predict_Final)
        print("Predict: ", NN.predict(X_Predict_Final))
        # print("Storing model and weights")
        # NN.save_model(json_filename, h5_filename)
        print("Done!")
        KerasBack.clear_session()
        print("Cleared Keras!")

class ProcessTweetsGloveOnePassParam:
    def __init__(self, labeled_tweets_filename, embedding_filename):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename

    def process(self, json_filename, h5_filename):
        if (gf.Exists('/tmp/logs')):
            gf.DeleteRecursively('/tmp/logs')
        np.random.seed(11)
        # open the file with tweets
        X_all = []
        Y_all = []
        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter = ',')
            for r in csv_file:
                if i !=0:
                    tweet = r[0]
                    label = r[1]
                    X_all.append(tweet)
                    Y_all.append(label)
                i = i + 1
        print("Data Ingested")
        # divide the data into training and test
        num_data = len(X_all)
        limit = math.ceil(num_data * 0.60)
        X_train_sentences = X_all
        Y_train = Y_all
        G = GloveEmbedding(self.embedding_filename)
        word_to_idx, idx_to_word, embedding = G.read_embedding()
        S = SentenceToIndices(word_to_idx)
        X_train_indices, max_len = S.map_sentence_list(X_train_sentences)
        print("Train data mappend to indices")
        P = PadSentences(max_len)
        X_train_pad = P.pad_list(X_train_indices)
        print("Train data padded")
        #convert to numPY arrays
        X_train = np.array(X_train_pad)
        Y_train = np.array(Y_train)
        Y_train = to_categorical(Y_train, num_classes=3)
        print("Train data convert to numpy arrays")
        model = KerasClassifier(build_fn=TweetSentiment2LSTMMaxDenseSequential(max_len, G).build(first_layer_units=50,
                                first_layer_dropout=0.4, second_layer_units=50, second_layer_dropout=0.5,
                                relu_dense_layer=64, dense_layer_units=3), verbose=0)
        # define the grid search parameters
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        grid = GridSearchCV(estimator=model, param_grid=param_grid )
        grid_result = grid.fit(X_train, Y_train)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))