from datetime import datetime
from operator import itemgetter

from keras import backend as KerasBack
from keras.callbacks import History
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from ths.nn.metrics.multiple import *
from ths.nn.sequences.tweets_cnn import *
from ths.nn.sequences.ml_process.tweets_processor import TweetsProcessor


class BestModelsMulticlassCNN(TweetsProcessor):
    def process_neural_network(self, max_len, g):
        self.nn = None
        models = list()

        self.nn = TweetSentimentInceptionOneChan(max_len, g)

        class_weight_val = class_weight.compute_class_weight('balanced', np.unique(self.y_all), self.y_all)
        class_weight_dictionary = {0: class_weight_val[0], 1: class_weight_val[1], 2: class_weight_val[2]}

        self.nn.build(filters=11, padding='valid', dense_units=16)
        self.nn.summary()

        # Assign the parameters agree the optimizer to use
        params_compile = {}

        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        params_compile['optimizer'] = rmsprop

        self.nn.compile(loss="categorical_crossentropy", metrics=['accuracy', precision, recall, f1, fprate],
                        **params_compile)

        history = History()

        # Assign batch size to fit function
        params_fit = {}
        params_fit['batch_size'] = 32

        self.nn.fit(self.x_train, self.y_train, epochs=20, callbacks=[history],
                    class_weight=class_weight_dictionary, **params_fit)

        predicted = self.nn.predict(self.x_valid)
        predicted = np.argmax(predicted, axis=1)

        # Getting metrics
        acc = accuracy(self.y_valid, predicted)
        c_matrix = confusion_matrix(self.y_valid, predicted)
        track = ("Confusion matrix : \n{}\n"
                 "Model accuracy: {}\n").format(c_matrix, acc)

        prec_1, recall_1, f1_1, spec_1, track = self.calculate_cm_metrics(c_matrix, track)

        print("PREC_1: ", prec_1)
        print("RECALL_1: ", recall_1)
        print("F1_1: ", f1_1)
        print("SPEC_1: ", spec_1)
        print("TRACK: ", track)

        KerasBack.clear_session()
        return models
