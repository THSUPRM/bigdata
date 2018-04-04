import numpy as np
np.random.seed(10)
from ths.utils.sentences import SentenceToEmbedding

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.regularizers import l2

np.random.seed(1)

class TweetSentiment2LSTM:

    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None
        self.sentiment_map = {0 : 'negative', 1 : 'positive', 2: 'neutral'}

    def build(self, first_layer_units = 128, first_layer_dropout=0.5, second_layer_units = 128,
              second_layer_dropout = 0.5, dense_layer_units = 2):

        # Input Layer
        sentence_input = Input(shape = (self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        #X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1") (X)
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
        word_to_idx, idx_to_word, word_embeddings = self.embedding_builder.read_embedding()
        #vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_embeddings
        #embedding_matrix = np.vstack([word_embeddings, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False, name="EMBEDDING")
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
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)
        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1") (X)
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
        word_to_idx, idx_to_word, word_embeddings = self.embedding_builder.read_embedding()
        #vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_embeddings
        #embedding_matrix = np.vstack([word_embeddings, np.zeros((vocabulary_len,))])

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
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # First LSTM Layer
        # X  = LSTM(first_layer_units, return_sequences=True, name='LSTM_1', kernel_regularizer=l2(0.1))(embeddings)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)

        # Dropout regularization
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
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

    def build(self, first_layer_units = 128, first_layer_dropout=0.3, second_layer_units = 128,
              second_layer_dropout = 0.5, third_layer_units = 128, third_layer_dropout = 0.5,
              relu_dense_layer = 30, dense_layer_units = 1):
        # Input Layer
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        X = LSTM(first_layer_units, return_sequences=True, name='LSTM_1')(embeddings)
        #X = Bidirectional(LSTM(first_layer_units, return_sequences=True, name='LSTM_1'))(embeddings)
        X = Dropout(first_layer_dropout, name="DROPOUT_1")(X)
        X = Dense(60, kernel_initializer='normal',  activation='relu')(X)
        X = LSTM(second_layer_units, return_sequences=False, name="LSTM_2")(X)
        X = Dropout(second_layer_dropout, name="DROPOUT_2")(X)
        X = Dense(relu_dense_layer, activation='relu')(X)
        X = Dense(dense_layer_units)(X)
        X = Activation("sigmoid", name="SIGMOID_1")(X)
        self.model = Model(input=sentence_input, output=X)