from datetime import datetime
from operator import itemgetter

from keras import backend as KerasBack
from keras.callbacks import History
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from ths.nn.metrics.multiple import *
from ths.nn.sequences.tweets import *
from ths.nn.sequences.ml_process.tweets_processor import TweetsProcessor


class SequentialModelBestGRU(TweetsProcessor):
    def process_neural_network(self, max_len, g):
        self.nn = TweetSentiment2LSTMMaxDenseSequential(max_len, g)
        models = list()

        class_weight_val = class_weight.compute_class_weight('balanced', np.unique(self.y_all), self.y_all)
        class_weight_dictionary = {0: class_weight_val[0], 1: class_weight_val[1], 2: class_weight_val[2]}

        for combination in self.params:
            file_name = self.route_files + "/model" + str(combination).replace(" ", "") + ".txt"
            log = open(file_name, "a+")
            start_time_comb = datetime.now()
            log.write("Start time: " + str(start_time_comb) + "\n")
            log.write("\nCOMBINATION: " + str(combination) + "\n")

            l = list(combination)

            desc = ""

            self.nn.build(layer_units_1=l[4], kernel_reg_1=l[5], recu_dropout_1=l[6], dropout_1=l[7],
                                 layer_units_2=l[8], kernel_reg_2=l[9], recu_dropout_2=l[10], dropout_2=l[11],
                                 dense_layer_1=l[12], regula_dense_1=l[13], dense_layer_2=l[14], attention=True)

            self.nn.summary()

            # Assign the parameters agree the optimizer to use
            params_compile = {}
            if l[15] == 'RMSPROP':
                rmsprop = RMSprop(lr=l[0], rho=0.9, epsilon=1e-06)
                params_compile['optimizer'] = rmsprop
                desc = desc + "\nRMSPROP with learning rate: " + str(l[0]) + " rho: 0.9 and epsilon=1e-06"

            self.nn.compile(loss="categorical_crossentropy", metrics=['accuracy', precision, recall, f1, fprate],
                            **params_compile)

            history = History()
            desc = desc + "\nEpochs: " + str(l[2]) + " Batch Size: " + str(l[3])

            # Assign batch size to fit function
            params_fit = {}
            if l[3] != 0:
                params_fit['batch_size'] = l[3]

            self.nn.fit(self.x_train, self.y_train, epochs=l[2], callbacks=[history],
                        class_weight=class_weight_dictionary,
                        **params_fit)

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
                     file_name, desc]

            models.append(model)
            models = sorted(models, key=itemgetter(3, 2, 1))

            KerasBack.clear_session()
            log.write(track)
            log.write(
                ("\nExecution time: {} minutes".format(((datetime.now() - start_time_comb).total_seconds()) / 60)))
            log.write(("\nFinish time: " + str(datetime.now())))
            log.close()
        return models
