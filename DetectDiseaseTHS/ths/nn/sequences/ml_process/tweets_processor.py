import csv
import itertools
import math
from abc import abstractmethod
from datetime import datetime

import numpy as np
from keras.utils import to_categorical

from ths.nn.sequences.tweets import *
from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, PadSentences, TrimSentences


class TweetsProcessor:
    def __init__(self, labeled_tweets_filename, embedding_filename, optimizer):
        self.labeled_tweets_filename = labeled_tweets_filename
        self.embedding_filename = embedding_filename
        self.optimizer = optimizer
        self.start_time_comp = datetime.now()
        np.random.seed(11)

    @abstractmethod
    def process_neural_network(self):
        pass

    def process(self):
        self.num_params = 16
        self.params = self.get_best_models_params()

        self.load_data()
        x_train_pad, max_len, g = self.get_glove_embedding()
        self.nn = TweetSentiment2LSTMHyper(max_len, g)

        self.get_partitioned_sets(x_train_pad)
        self.define_and_save_dictionary_datasets()
        models = self.process_neural_network()
        self.save_best_models_file(models)

    def load_data(self):
        self.x_all = []
        self.y_all = []
        self.all_data = []

        with open(self.labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
            i = 0
            csv_file = csv.reader(f, delimiter=',')
            for r in csv_file:
                if i != 0:
                    self.all_data.append(r)
                i = i + 1
        # print("len(All): ", len(all_data))
        # np.random.shuffle(all_data)

        zeros_count = 0
        ones_count = 0
        twos_count = 0

        for r in self.all_data:
            tweet = r[0].strip()
            label = int(r[1])
            self.x_all.append(str(tweet).strip())
            self.y_all.append(label)
            if label == 0:
                zeros_count += 1
            elif label == 1:
                ones_count += 1
            elif label == 2:
                twos_count += 1
        # print("LEN X_all:" , len(X_all))
        # print("LEN y_all:", len(y_all))
        # print("y_all ZEROS: ", zeros_count)
        # print("y_all ONES: ", ones_count)
        # print("y_all TWOS: ", twos_count)

    def get_glove_embedding(self):
        g = GloveEmbedding(self.embedding_filename, dimensions=50)
        word_to_idx, idx_to_word, embedding = g.read_embedding()
        s = SentenceToIndices(word_to_idx)
        x_train_indices, max_len = s.map_sentence_list(self.x_all)

        if max_len % 2 != 0:
            max_len = max_len + 1

        p = PadSentences(max_len)
        x_train_pad = p.pad_list(x_train_indices)

        # TRIM Tweets to remove noisy data
        trim_size = max_len
        trim = TrimSentences(trim_size)
        x_train_pad = trim.trim_list(x_train_pad)

        return x_train_pad, max_len, g

    def get_partitioned_sets(self, x_train_pad):
        self.x_mapped = np.array(x_train_pad)
        self.y_mapped = np.array(self.y_all)

        self.y_train_old = self.y_mapped
        self.y_mapped = to_categorical(self.y_mapped, num_classes=3)

        num_data = len(self.x_all)

        test_count = math.floor(num_data * 0.20)
        # TEST SET -> data set for test the model
        self.x_test = self.x_mapped[0:test_count]
        self.y_test = self.y_train_old[0:test_count]

        # print("Len:" + str(len(x_test)) + " ARRAY:" + str(x_test))

        # CROSS VALIDATION SET -> data set for validate the model
        self.x_valid = self.x_mapped[(test_count + 1):(test_count * 2)]
        self.y_valid = self.y_train_old[(test_count + 1):(test_count * 2)]

        # print("Len:" + str(len(x_valid)) + " ARRAY:" + str(x_valid))

        # TRAINING SET -> data set for training and cross validation
        self.x_train = self.x_mapped[(test_count * 2) + 1:]
        self.y_train = self.y_mapped[(test_count * 2) + 1:]

        # print("Len:" + str(len(x_train)) + " ARRAY:" + str(x_train))
        # print("y_train LEN: ", len(y_train))
        # zeros_count = 0
        # ones_count = 0
        # twos_count = 0
        # for x in y_train:
        #     # print("x[0] x[1] x[2]: ", x[0], x[1], x[2])
        #     if x[0] == 1:
        #         zeros_count += 1
        #     elif x[1] == 1:
        #         ones_count += 1
        #     elif x[2] == 1:
        #         twos_count += 1
        # print("LEN y_train:", len(y_train))
        # print("y_train ZEROS: ", zeros_count)
        # print("y_train ONES: ", ones_count)
        # print("y_train TWOS: ", twos_count)

    def save_dataset_file(self, route, data):
        file = open(route, "w")
        file.write("\n".join([str(a) for a in data]))
        file.close()

    def define_and_save_dictionary_datasets(self):
        sets = {
            "models/RNN/x_test.txt": self.x_test,
            "models/RNN/y_test.txt": self.y_test,
            "models/RNN/x_validation.txt": self.x_valid,
            "models/RNN/y_validation.txt": self.y_valid,
            "models/RNN/x_training.txt": self.x_train,
            "models/RNN/y_training.txt": self.y_train
        }
        for route, data in sets.items():
            self.save_dataset_file(route, data)

    def get_best_models_params(self):
        return [  # (0.003,0,5,32,50,0,0,0.1,50,0,0,0.1,64,0,3,'RMSPROP')
            (0.003, 0, 20, 32, 50, 0, 0, 0.1, 50, 0, 0, 0.1, 64, 0, 3, 'RMSPROP')
            , (0.001, 0, 60, 32, 50, 0, 0, 0.5, 50, 0, 0, 0.5, 64, 0, 3, 'RMSPROP')
            , (0.001, 0, 5, 32, 50, 0, 0, 0.3, 50, 0, 0, 0.1, 32, 0, 3, 'RMSPROP')
            , (0.001, 0, 10, 32, 50, 0, 0, 0.3, 50, 0, 0, 0.1, 32, 0, 3, 'RMSPROP')
            , (0.001, 0, 20, 32, 50, 0, 0, 0.5, 50, 0, 0, 0.5, 32, 0, 3, 'RMSPROP')
            , (0.001, 0, 60, 32, 50, 0, 0, 0.3, 50, 0, 0, 0.1, 64, 0, 3, 'RMSPROP')
        ]
        # longi = sum(1 for x in params)
        # print("LONGITUD PARAMS: " + str(longi))
        # for combination in params:
        #     print(type(combination))
        #     print(str(combination).replace(" ", ""))

    def calculate_cm_metrics(self, c_matrix, track):
        prec_0 = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[1][0] + c_matrix[2][0])
        prec_1 = c_matrix[1][1] / (c_matrix[1][1] + c_matrix[0][1] + c_matrix[2][1])
        prec_2 = c_matrix[2][2] / (c_matrix[2][2] + c_matrix[1][2] + c_matrix[0][2])

        recall_0 = c_matrix[0][0] / (c_matrix[0][0] + c_matrix[0][1] + c_matrix[0][2])
        recall_1 = c_matrix[1][1] / (c_matrix[1][1] + c_matrix[1][0] + c_matrix[1][2])
        recall_2 = c_matrix[2][2] / (c_matrix[2][2] + c_matrix[2][0] + c_matrix[2][1])

        f1_0 = 2 * ((prec_0 * recall_0) / (prec_0 + recall_0))
        f1_1 = 2 * ((prec_1 * recall_1) / (prec_1 + recall_1))
        f1_2 = 2 * ((prec_2 * recall_2) / (prec_2 + recall_2))

        tn_0 = c_matrix[1][1] + c_matrix[1][2] + c_matrix[2][1] + c_matrix[2][2]
        tn_1 = c_matrix[0][0] + c_matrix[0][2] + c_matrix[2][0] + c_matrix[2][2]
        tn_2 = c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1]

        spec_0 = tn_0 / (tn_0 + c_matrix[1][0] + c_matrix[2][0])
        spec_1 = tn_1 / (tn_1 + c_matrix[0][1] + c_matrix[2][1])
        spec_2 = tn_2 / (tn_2 + c_matrix[0][2] + c_matrix[1][2])

        t = track + """ Precision 0: {} 
                        Precision 1: {}
                        Precision 2: {}
                        Recall 0: {}
                        Recall 1: {}
                        Recall 2: {}
                        F1 Score 0: {}
                        F1 Score 1: {}
                        F1 Score 2: {}
                        Specificity 0: {}
                        Specificity 1: {}
                        Specificity 2: {}""".format(prec_0, prec_1, prec_2, recall_0, recall_1, recall_2, f1_0, f1_1,
                                                    f1_2, tn_0, tn_1, tn_2, spec_0, spec_1, spec_2)
        return prec_1, recall_1, f1_1, spec_1, t

    def get_hyper_params_matrix(self, i=1):
        a = [
            ['learningRate', 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
            ['momentum', 0],
            ['epochs', 5, 10, 20, 40, 60],
            ['batchSize', 32],
            # LSTM1
            ['layerUnits1', 50],  # the same that the glove number
            ['kernelReg1', 0],
            ['recuDropout1', 0],
            # Dropout1
            ['dropout1', 0, 0.1, 0.3, 0.5],
            # LSTM2
            ['layerUnits2', 50],
            ['kernelReg1', 0],
            ['recuDropout2', 0],
            # Dropout2
            ['dropout2', 0, 0.1, 0.3, 0.5],
            # DenseLayer1
            ['denseLayer1', 32, 64],
            ['regulaDense1', 0],
            # DenseLayer2
            ['denseLayer2', 1],
            # Optimizer
            ['optimizer', self.optimizer]
        ]
        b = [a[n][1:] for n in range(0, i)]
        return itertools.product(*b)
        # params = [(0.001, 0, 5, 32, 50, 0, 0, 0.3, 50, 0, 0, 0.1, 32, 0, 1, 'RMSPROP')]

    def save_best_models_file(self, models):
        file = open("models/RNN/bestModels" + str(datetime.now())[:19].replace(" ", "T") + ".txt", "a+")

        for m in models:
            file.write("--------------------------------------------------------------------------------------------")
            file.write("\nacc: " + str(m[0]))
            file.write("\nprec_1: " + str(m[1]))
            file.write("\nrecall_1: " + str(m[2]))
            file.write("\nf1_1: " + str(m[3]))
            file.write("\nspec_1: " + str(m[4]))
            file.write("\nfileName: " + str(m[5]))
            file.write("\nmodel: " + str(m[6]))
            file.write("\n--------------------------------------------------------------------------------------------")

        file.write("\nStart TOTAL execution time: " + str(self.start_time_comp))
        file.write("\nTOTAL execution time: {} hours".format(
            ((datetime.now() - self.start_time_comp).total_seconds()) / (60 * 60)))
        file.write("\nFinish TOTAL execution time: " + str(datetime.now()))
        file.close()
