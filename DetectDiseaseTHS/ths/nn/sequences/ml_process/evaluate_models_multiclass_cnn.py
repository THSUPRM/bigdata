import itertools
import time
from datetime import datetime
from operator import itemgetter

from keras import backend as KerasBack
from keras.callbacks import History
from keras.optimizers import RMSprop, Adam, Adadelta
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from ths.nn.metrics.multiple import *
from ths.nn.sequences.tweets_cnn import *
from ths.nn.sequences.ml_process.tweets_processor import TweetsProcessor


class EvaluateModelsMulticlassCNN(TweetsProcessor):
    def process_neural_network(self, max_len, g):
        self.num_params = 7
        self.nn = None
        models = list()

        l = list()
        params = self.get_hyper_matrix_cnn(self.num_params)
        class_weight_val = class_weight.compute_class_weight('balanced', np.unique(self.y_all), self.y_all)
        class_weight_dictionary = {0: class_weight_val[0], 1: class_weight_val[1], 2: class_weight_val[2]}
        count_models = 0

        # params = [(0.03, 20, 32, 11, 0.3, 128, 'ADAM')]

        # for combination in itertools.islice(params, 0, 10):    # -> Basic test
        for combination in itertools.islice(params, 0, 360):     # -> 1st host
        # for combination in itertools.islice(params, 360, 720):  # -> 2nd host
        # for combination in itertools.islice(params, 720, 1080):  # -> 3th host
        # for combination in params:
            count_models += 1
            self.nn = TweetSentimentInceptionV2_3x3(max_len, g)
            file_name = str(self.route_files) + "/model" + str(combination).replace(" ", "") + ".txt"
            log = open(file_name, "a+")
            start_time_comb = datetime.now()
            log.write("Start time: " + str(start_time_comb) + "\n")
            log.write("COMBINATION: " + str(combination) + "\n")

            l = [0] * self.num_params
            for e in range(0, self.num_params):
                l[e] = combination[e]

            self.nn.build(filters=l[3], dropout=l[4], dense_units=l[5])
            self.nn.summary()

            # Assign the parameters agree the optimizer to use
            params_compile = {}
            if l[6] == 'RMSPROP':
                rmsprop = RMSprop(lr=l[0], rho=0.9, epsilon=1e-06)
                params_compile['optimizer'] = rmsprop
            elif l[6] == 'ADAM':
                adam = Adam(lr=l[0], beta_1=0.9, beta_2=0.999)
                params_compile['optimizer'] = adam
            elif l[6] == 'ADADELTA':
                adaDelta = Adadelta(lr=l[0], rho=0.95, epsilon=1e-08, decay=0.0)
                params_compile['optimizer'] = adaDelta
            else:
                print("OPTIMIZADOR: El optimizador a crear no esta en lista...")

            self.nn.compile(loss="categorical_crossentropy", metrics=['accuracy', precision, recall, f1, fprate],
                            **params_compile)

            history = History()

            self.nn.fit(self.x_train, self.y_train, epochs=l[1], batch_size=l[2], callbacks=[history],
                        class_weight=class_weight_dictionary)

            predicted = self.nn.predict(self.x_valid)
            predicted = np.argmax(predicted, axis=1)

            # Getting metrics
            acc = accuracy(self.y_valid, predicted)
            c_matrix = confusion_matrix(self.y_valid, predicted)
            track = ("Confusion matrix : \n{}\n"
                     "Model accuracy: {}\n").format(c_matrix, acc)

            prec_1, recall_1, f1_1, spec_1, track = self.calculate_cm_metrics(c_matrix, track)

            model = [acc,  # Accuracy
                     prec_1,  # Precision
                     recall_1,  # Recall
                     f1_1,  # F1 Score
                     spec_1,  # Specificity
                     file_name, 'NO']

            if prec_1 != 'nan' and recall_1 != 0.0:
                if len(models) < 8:
                    models.append(model)
                else:
                    models.append(model)
                    models = sorted(models, key=itemgetter(3, 2, 1))
                    # print("Will erase: " + str(models[3]))
                    models.pop(3)

                # SAVE MODEL
            json_route = self.route_files + "/model" + str(combination).replace(" ", "") + ".json"
            h5_route = self.route_files + "/model" + str(combination).replace(" ", "") + ".h5"
            self.nn.save_model(json_route, h5_route)
            print("Saved model to disk")

            KerasBack.clear_session()
            log.write(track)
            log.write(
                ("\nExecution time: {} minutes".format(((datetime.now() - start_time_comb).total_seconds()) / 60)))
            log.write(("\nFinish time: " + str(datetime.now())))
            log.close()
            print("******************************************************************MODEL EVALUATED #: ", count_models)
        return models
