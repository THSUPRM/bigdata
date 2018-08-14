import numpy as np

np.random.seed(10)

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization, Embedding, GRU
from keras.layers import Permute, Multiply, TimeDistributed
from keras.optimizers import RMSprop
from keras.regularizers import l2
from datetime import datetime

np.random.seed(1)


class TweetSentiment2LSTM:

    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = Model
        self.sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}

    def build(self, first_layer_units=128, first_layer_dropout=0.5, second_layer_units=128,
              second_layer_dropout=0.5, dense_layer_units=2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(input)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(input)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="dropout")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        # Send to a Dense Layer with softmax activation
        X = Dense(dense_layer_units, name="DENSE_1")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def summary(self):
        self.model.summary()

    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_input = self.embedding_builder.read_embedding()
        # vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_input
        # embedding_matrix = np.vstack([word_input, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False,
                                    name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def pretrained_embedding_layer_seq(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_input = self.embedding_builder.read_embedding()
        # vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_input
        # embedding_matrix = np.vstack([word_input, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension,
                                    input_length=self.max_sentence_len, trainable=False, name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def compile(self, loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, Y, epochs=50, batch_size=32, shuffle=True, callbacks=None, validation_split=0.0,
            class_weight=None):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight)

    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)

    def predict(self, X):
        return self.model.predict(X)

    def get_sentiment(self, prediction):
        return np.argmax(prediction)

    def sentiment_string(self, sentiment):
        return self.sentiment_map[sentiment]

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return


class TweetSentiment3LSTM:

    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None
        self.sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}

    def build(self, first_layer_units=128, first_layer_dropout=0.5, second_layer_units=128,
              second_layer_dropout=0.5, third_layer_units=128, third_layer_dropout=0.5, dense_layer_units=2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        # First LSTM Layer
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(input)
        # Dropout regularization
        X = Dropout(first_layer_dropout, name="dropout")(X)
        # Second LSTM Layer
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_2')(X)
        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_2")(X)
        # Third layer
        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_3")(X)
        # Third Layer Dropout
        X = Dropout(second_layer_dropout, name="DROPOUT_3")(X)
        # Send to a Dense Layer with softmax activation
        X = Dense(dense_layer_units, name="DENSE_1")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)

    def summary(self):
        self.model.summary()

    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_input = self.embedding_builder.read_embedding()
        # vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_input
        # embedding_matrix = np.vstack([word_input, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False,
                                    name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, Y, epochs=50, batch_size=32, shuffle=True, validation_split=0.0):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, validation_split=validation_split)

    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)

    def predict(self, X):
        return self.model.predict(X)

    def get_sentiment(self, prediction):
        return np.argmax(prediction)

    def sentiment_string(self, sentiment):
        return self.sentiment_map[sentiment]

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return


class TweetSentiment2LSTM2Dense(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units=128, first_layer_dropout=0.5, second_layer_units=128,
              second_layer_dropout=0.5, third_layer_units=128, third_layer_dropout=0.5,
              relu_dense_layer=64, dense_layer_units=2):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(input)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(input)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="dropout")(X)
        # Second LSTM Layer
        # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        X = Dense(relu_dense_layer, name="DENSE_1", activation='relu')(X)
        # Send to a Dense Layer with sigmoid activation
        X = Dense(dense_layer_units, name="DENSE_2")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model = Model(input=sentence_input, output=X)


class TweetSentiment2LSTMHyper(TweetSentiment2LSTM):
    model_created = ""

    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def get_model(self):
        return self.model

    def build(self, layer_units_1=0, kernel_reg_1=0, recu_dropout_1=0, dropout_1=0, layer_units_2=0, kernel_reg_2=0,
              recu_dropout_2=0, dropout_2=0, dense_layer_1=0, regula_dense_1=0, dense_layer_2=0, attention=False):

        self.model_created = ""

        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        X = None

        print("LEN INPUT: ", input.get_shape())

        # LSTM 1
        X = self.create_LSTM(input, layer_units_1, kernel_reg_1, recu_dropout_1, True, 'LSTM_1')
        # X = GRU(72, layer_units_1, input_shape=50, return_sequences=True, activation='tanh', name='GRU')(input)
        # Dropout 1
        X = self.create_dropout(X, dropout_1, 'dropout_1')
        # Attention
        if attention:
            X = self.create_attention(X)
        # LSTM 2
        X = self.create_LSTM(X, layer_units_2, kernel_reg_2, recu_dropout_2, False, 'LSTM_2')
        # X = GRU(X, layer_units_2, return_sequences=False, activation='tanh', name='GRU')
        # Dropout 2
        X = self.create_dropout(X, dropout_2, 'dropout_2')
        # Dense Layer 1
        X = self.create_dense(X, dense_layer_1, regula_dense_1, True, 'tanh', 'dense_1')
        # Dense Layer 2
        X = self.create_dense(X, dense_layer_2, 0, False, None, 'dense_2')

        # X = BatchNormalization()(X)
        # self.model_created = self.model_created + "\nBatch Normalization"

        X = self.activation(X, dense_layer_2, 'act')

        self.model = Model(input=sentence_input, output=X)
        return self.model_created

    def create_LSTM(self, input, layer_units, kernel_reg, recu_dropout, return_sequences, name):
        params = {}
        try:
            if layer_units == 0 and kernel_reg == 0 and recu_dropout == 0:
                self.model_created = self.model_created + "Error, you need to assign values to layer_units to initialize the LSTM layer"
                raise Exception("ERROR creating LSTM layer all params are 0")
            else:
                if kernel_reg != 0:
                    params['kernel_regularizer'] = l2(kernel_reg)
                if recu_dropout != 0:
                    params['recurrent_dropout'] = recu_dropout

                X = LSTM(layer_units, return_sequences=return_sequences, name=name, **params)(input)
                self.model_created = self.model_created + "\n" + name + " with layer_units: " + str(layer_units) + \
                                     " kernel_regularizer: l2(" + str(kernel_reg) + ") recurrent_dropout: " + \
                                     str(recu_dropout) + " and return sequences: " + str(return_sequences)
        except Exception as e:
            print(e)
            print("ERROR creating LSTM layer in the model")
        return X

    def create_dropout(self, input, dropout, name):
        try:
            X = Dropout(dropout, name=name)(input)
            self.model_created = self.model_created + "\nDropout: " + str(dropout)
        except Exception as e:
            print(e)
            print("ERROR creating Dropout layer in the model")
        return X

    def create_dense(self, input, dense_layer, regula_dense, activation, type_activation, name):
        try:
            if dense_layer == 0 and regula_dense == 0:
                self.model_created = self.model_created + "Error, you need to assign values to dense layer units and regularization to the dense layer"
                raise Exception("ERROR creating DENSE layer all params are 0")
            else:
                params = {}
                if regula_dense != 0:
                    params['kernel_regularizer'] = l2(regula_dense)
                if activation:
                    params['activation'] = type_activation

                X = Dense(dense_layer, **params, name=name)(input)
                self.model_created = self.model_created + "\nDense: " + str(dense_layer) + " kernel_regularizer: l2(" + \
                                     str(regula_dense) + ") activation: " + str(type_activation)
        except Exception as e:
            print(e)
            print("ERROR creating Dense layer in the model")
        return X

    def activation(self, input, dense_layer_2, name):
        try:
            if dense_layer_2 == 1:
                X = Activation("sigmoid", name=name)(input)
                self.model_created = self.model_created + "\nActivation type: Sigmoid"
            elif dense_layer_2 > 1:
                X = Activation("softmax", name=name)(input)
                self.model_created = self.model_created + "\nActivation type: Softmax"
        except Exception as e:
            print(e)
            print("ERROR setting the activation layer in the model")
        return X

    def create_attention(self, input):
        try:
            attention = Permute((2, 1), name="Attention_Permute")(input)
            attention = TimeDistributed(Dense(self.max_sentence_len, activation='softmax', name="Attention_Dense"))(
                attention)
            attention_probs = Permute((2, 1), name='attention_probs')(attention)

            output_attention_mul = Multiply(name='attention_multiplu')([input, attention_probs])
        except Exception as e:
            print(e)
            print("ERROR setting the attention layer in the model")
        return output_attention_mul


class TweetSentiment2LSTMMaxDenseBidirectional(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, layer_units_1=0, kernel_reg_1=0, recu_dropout_1=0, dropout=0, dense_layer_1=0,
              layer_units_2=0, kernel_reg_2=0, recu_dropout_2=0, dropout_2=0, dense_layer_2=0, dense_last=3):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        X = Bidirectional(LSTM(layer_units_1, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(kernel_reg_1),
                               recurrent_dropout=recu_dropout_1))(input)
        # X = LSTM(layer_units_1, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(kernel_reg_1), recurrent_dropout=recu_dropout_1)(input)
        X = Dropout(dropout, name="dropout")(X)
        X = Dense(dense_layer_1, activation='relu', kernel_regularizer=l2(kernel_reg_1))(X)

        X = LSTM(layer_units_2, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(kernel_reg_2),
                 recurrent_dropout=recu_dropout_2)(X)
        X = Dropout(dropout_2, name="DROPOUT_2")(X)
        X = Dense(dense_layer_2, activation='relu', kernel_regularizer=l2(kernel_reg_2))(X)
        X = Dense(dense_last)(X)
        X = BatchNormalization()(X)
        X = Activation("softmax", name="softmax_final")(X)
        self.model = Model(input=sentence_input, output=X)


class TweetSentiment2LSTMMaxDenseSequential(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, layer_units_1=0, kernel_reg_1=0, recu_dropout_1=0, dropout_1=0, layer_units_2=0, kernel_reg_2=0,
              recu_dropout_2=0, dropout_2=0, dense_layer_1=0, regula_dense_1=0, dense_layer_2=0, attention=False):
        # Embedding layer
        input_layer = self.pretrained_embedding_layer_seq()
        model = Sequential()
        model.add(input_layer)

        model.add(LSTM(layer_units_1, return_sequences=True, name='LSTM_1'))
        # model.add(GRU(layer_units_1, return_sequences=True))
        model.add(Dropout(dropout_1, name="dropout"))
        # model.add(Dense(dense_layer_1, activation='tanh'))
        # model.add(LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2))

        # model.add(Permute((2, 1), name="Attention_Permute"))
        # model.add(TimeDistributed(Dense(self.max_sentence_len, activation='softmax', name="Attention_Dense")))
        # model.add(Permute((2, 1), name="attention_probs"))
        # model.add(Multiply(name='attention_multiplu', [X, attention_probs]))

        # model.add(LSTM(layer_units_2, return_sequences=False, name="LSTM_2"))
        model.add(GRU(layer_units_2, return_sequences=False))
        model.add(Dropout(dropout_2, name="DROPOUT_2"))
        model.add(Dense(dense_layer_1, activation='tanh'))
        model.add(Dense(dense_layer_2))
        model.add(Activation("softmax", name="softmax_final"))
        self.model = model
