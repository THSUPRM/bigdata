import numpy as np
np.random.seed(10)

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, BatchNormalization, Embedding
from keras.optimizers import RMSprop
from keras.regularizers import l2
from datetime import datetime

np.random.seed(1)

class TweetSentiment2LSTM:

    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = Model
        self.sentiment_map = {0 : 'negative', 1 : 'positive', 2: 'neutral'}

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, dense_layer_units = 2):

        # Input Layer
        sentence_input = Input(shape = (self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        # First LSTM Layer
        #X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(input)
        X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(input)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="dropout") (X)
        # Second LSTM Layer
       # X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(0.1))(X)
        # Second Layer Dropout
        X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        # Send to a Dense Layer with softmax activation
        X = Dense(dense_layer_units,name="DENSE_1")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model  = Model(input = sentence_input, output=X)
    def summary(self):
        self.model.summary()

    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_input = self.embedding_builder.read_embedding()
        #vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_input
        #embedding_matrix = np.vstack([word_input, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False, name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def pretrained_embedding_layer_seq(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_input = self.embedding_builder.read_embedding()
        #vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_input
        #embedding_matrix = np.vstack([word_input, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, input_length=self.max_sentence_len, trainable=False, name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split)

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
        self.sentiment_map = {0 : 'negative', 1 : 'positive', 2: 'neutral'}

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5, dense_layer_units = 2):

        # Input Layer
        sentence_input = Input(shape = (self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        # First LSTM Layer
        X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(input)
        # Dropout regularization
        X = Dropout(first_layer_dropout, name="dropout") (X)
        # Second LSTM Layer
        X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_2')(X)
        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_2") (X)
        # Third layer
        X  = LSTM(second_layer_units, return_sequences=False, name="LSTM_3")(X)
        # Third Layer Dropout
        X = Dropout(second_layer_dropout, name="DROPOUT_3")(X)
        # Send to a Dense Layer with softmax activation
        X = Dense(dense_layer_units,name="DENSE_1")(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        # create the model
        self.model  = Model(input = sentence_input, output=X)
    def summary(self):
        self.model.summary()

    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_input = self.embedding_builder.read_embedding()
        #vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_input
        #embedding_matrix = np.vstack([word_input, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False, name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, validation_split=0.0):
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

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 64, dense_layer_units = 2):
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

class TweetSentiment2LSTMMaxDense(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, layer_units_1=0, kernel_reg_1=0, recu_dropout_1=0, dropout_1=0, layer_units_2=0, kernel_reg_2=0,
              recu_dropout_2=0, dropout_2=0, dense_layer_1=0, regula_dense_1=0, dense_layer_2=0):

        # Create file to save models
        filename = "model-" + str(datetime.now()).replace(" ", "-")[:19] + ".txt"
        f = open("models/" + filename, "a+")

        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        input_layer = self.pretrained_embedding_layer()
        input = input_layer(sentence_input)
        X = None

        # LSTM 1
        X = self.create_LSTM(input, f, layer_units_1, kernel_reg_1, recu_dropout_1, True, 'LSTM_1')
        # Dropout 1
        X = self.create_dropout(X, f, dropout_1, 'dropout_1')
        # LSTM 2
        X = self.create_LSTM(X, f, layer_units_2, kernel_reg_2, recu_dropout_2, False, 'LSTM_2')
        # Dropout 2
        X = self.create_dropout(X, f, dropout_2, 'dropout_2')
        # Dense Layer 1
        X = self.create_dense(X, f, dense_layer_1, regula_dense_1, True, 'relu', 'dense_1')
        #Dense Layer 2
        X = self.create_dense(X, f, dense_layer_2, 0, False, None, 'dense_2')

        X = BatchNormalization()(X)
        X = Activation("softmax", name="softmax_final")(X)
        self.model = Model(input=sentence_input, output=X)

        f.close()
        return filename

    def create_dense(self, X, f, dense_layer, regula_dense, activation, type_activation, name):
        try:
            if dense_layer == 0 and regula_dense == 0:
                f.write("Error, you need to assign values to dense layer units and regularization to the dense layer")
                raise Exception("ERROR creating DENSE layer all params are 0")
            else:
                params = {}
                if regula_dense != 0:
                    params['kernel_regularizer'] = l2(regula_dense)
                if activation:
                    params['activation'] = type_activation

                X = Dense(dense_layer, **params, name=name)(X)
                f.write("\nDense: " + str(dense_layer) + " kernel_regularizer: l2(" + str(regula_dense) +
                        ") activation: " + type_activation)
                print("ENTRO DENSE LAYER 2")
        except Exception as e:
            print(e)
            print("ERROR creating Dense layer in the model")
        return X

    def create_dropout(self, X, f, dropout, name):
        try:
            if dropout == 0:
                f.write("Error, you need to assign values to dropout to the dropout layer")
                raise Exception("ERROR creating DROPOUT layer all params are 0")
            else:
                X = Dropout(dropout, name=name)(X)
                f.write("\nDropout: " + str(dropout))
                print("ENTRO DROPOUT")
        except Exception as e:
            print(e)
            print("ERROR creating Dropout layer in the model")
        return X

    def create_LSTM(self, input, f, layer_units, kernel_reg, recu_dropout, return_sequences, name):
        params = {}
        try:
            if layer_units == 0 and kernel_reg == 0 and recu_dropout == 0:
                f.write("Error, you need to assign values to layer_units to initialize the LSTM layer")
                raise Exception("ERROR creating LSTM layer all params are 0")
            else:
                if kernel_reg != 0:
                    params['kernel_regularizer'] = l2(kernel_reg)
                if recu_dropout != 0:
                    params['recurrent_dropout'] = recu_dropout

                X = LSTM(layer_units, return_sequences=return_sequences, name=name, **params)(input)
                f.write(name + " with layer_units: " + str(layer_units) + " kernel_regularizer: l2(" + str(
                    kernel_reg) + ") recurrent_dropout: " + str(recu_dropout) + "and return sequences: " + str(
                    return_sequences))
                print("ENTRO LSTM")
        except Exception as e:
            print(e)
            print("ERROR creating LSTM layer in the model")
        return X


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
        X = Bidirectional(LSTM(layer_units_1, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(kernel_reg_1), recurrent_dropout=recu_dropout_1))(input)
        # X = LSTM(layer_units_1, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(kernel_reg_1), recurrent_dropout=recu_dropout_1)(input)
        X = Dropout(dropout, name="dropout")(X)
        X = Dense(dense_layer_1, activation='relu', kernel_regularizer=l2(kernel_reg_1))(X)

        X = LSTM(layer_units_2, return_sequences=False, name="LSTM_2", kernel_regularizer=l2(kernel_reg_2), recurrent_dropout=recu_dropout_2)(X)
        X = Dropout(dropout_2, name="DROPOUT_2")(X)
        X = Dense(dense_layer_2, activation='relu', kernel_regularizer=l2(kernel_reg_2))(X)
        X = Dense(dense_last)(X)
        X = BatchNormalization()(X)
        X = Activation("softmax", name="softmax_final")(X)
        self.model = Model(input=sentence_input, output=X)


class  TweetSentiment2LSTMMaxDenseSequential(TweetSentiment2LSTM):
    def __init__(self, max_sentence_len, embedding_builder):
        super().__init__(max_sentence_len, embedding_builder)

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, relu_dense_layer = 64, dense_layer_units = 3, l2=None):
        # Embedding layer
        input_layer = self.pretrained_embedding_layer_seq()
        model = Sequential()
        model.add(input_layer)
        model.add(LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2, recurrent_dropout=0.4))
        model.add(Dropout(first_layer_dropout, name="dropout"))
        model.add(Dense(200, activation='relu', kernel_regularizer=l2))
        model.add(LSTM(second_layer_units, return_sequences=False, name="LSTM_2", kernel_regularizer=l2))
        model.add(Dropout(second_layer_dropout, name="DROPOUT_2"))
        model.add(Dense(relu_dense_layer, activation='relu', kernel_regularizer=l2))
        model.add(Dense(dense_layer_units))
        model.add(Activation("softmax", name="softmax_final"))
        self.model = model
        model.compile(optimizer=RMSprop(decay=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
        return model

