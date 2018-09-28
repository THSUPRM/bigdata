import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Reshape, Concatenate, ZeroPadding2D, Dropout
from keras.layers.embeddings import Embedding
from abc import abstractmethod
np.random.seed(7)


class  TweetSentiment2DCNN2Channel:
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None

    @abstractmethod
    def build(self, filters=4, dense_units=64, dropout=0):
        pass

    def pretrained_embedding_layer(self):
        # create Keras embedding layer
        word_to_idx, idx_to_word, word_embeddings = self.embedding_builder.read_embedding()
        # vocabulary_len = len(word_to_idx) + 1
        vocabulary_len = len(word_to_idx)
        emb_dimension = self.embedding_builder.get_dimensions()
        # get the matrix for the sentences
        embedding_matrix = word_embeddings
        # embedding_matrix = np.vstack([word_embeddings, np.zeros((vocabulary_len,))])

        # embedding layer
        embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, trainable=False,
                                    name="EMBEDDING")
        embedding_layer.build((None,))
        embedding_layer.set_weights([embedding_matrix])
        return embedding_layer

    def summary(self):
        self.model.summary()

    def compile(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit(self, X, Y, epochs = 50, batch_size = 32, shuffle=True, callbacks=None, validation_split=0.0, class_weight=None):
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks,
                       validation_split=validation_split, class_weight=class_weight, verbose=2)

    def evaluate(self, X_test, Y_test):
        return self.model.evaluate(X_test, Y_test)

    def predict(self, X):
        return self.model.predict(X)

    def get_sentiment(self, prediction):
        return np.argmax(prediction)

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return

    def get_inception_model(self, embeddings, filters, count, strides_level, n):
        # Branch No. 1
        branch_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_1X1_"+str(count))(embeddings)
        branch_1 = Conv2D(filters, kernel_size=(1, n), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_2_1X"+str(n)+"_"+str(count))(branch_1)
        branch_1 = Conv2D(filters, kernel_size=(n, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_2_"+str(n)+"X1_"+str(count))(branch_1)
        if strides_level == 1:
            branch_1 = Conv2D(filters * 2, kernel_size=(1, n), strides=(1, 1), padding='same', activation='relu',
                              name="CONV1_3_1X"+str(n)+"_"+str(count))(branch_1)
        elif strides_level == 2:
            branch_1 = Conv2D(filters * 2, kernel_size=(1, n), strides=(2, 2), padding='same', activation='relu',
                              name="CONV1_3_1X"+str(n)+"_"+str(count))(branch_1)
        else:
            print("!!!ERROR!!!: Strides number is WRONG in Branch #1....")
        branch_1 = Conv2D(filters * 2, kernel_size=(n, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_"+str(n)+"X1_"+str(count))(branch_1)

        # Branch No. 2
        branch_2 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV2_1X1_"+str(count))(embeddings)
        if strides_level == 1:
            branch_2 = Conv2D(filters * 2, kernel_size=(1, n), strides=(1, 1), padding='same', activation='relu',
                              name="CONV2_1X" + str(n) + "_" + str(count))(branch_2)
        elif strides_level == 2:
            branch_2 = Conv2D(filters * 2, kernel_size=(1, n), strides=(2, 2), padding='same', activation='relu',
                              name="CONV2_1X" + str(n) + "_" + str(count))(branch_2)
        else:
            print("!!!ERROR!!!: Strides number is WRONG in Branch #3....")
        branch_2 = Conv2D(filters, kernel_size=(n, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV2_" + str(n) + "X1_" + str(count))(branch_2)

        # Branch No. 3
        branch_3 = MaxPooling2D()
        if strides_level == 1:
            branch_3 = MaxPooling2D((1, 1), strides=(1, 1), padding='same', name="MAXPOL3_1X1_"+str(count))(embeddings)
        elif strides_level == 2:
            branch_3 = MaxPooling2D((1, 1), strides=(2, 2), padding='same', name="MAXPOL3_1X1_"+str(count))(embeddings)
        else:
            print("!!!ERROR!!!: Strides number is WRONG in Branch #3 Max Pooling....")
        branch_3 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV3_1X1_"+str(count))(branch_3)

        # Branch No. 4
        branch_4 = Conv2D(filters * 2, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV4_1X1_" + str(count))(embeddings)
        if strides_level == 1:
            branch_4 = Conv2D(filters * 2, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                              name="CONV4_1X1_" + str(count))(embeddings)
        elif strides_level == 2:
            branch_4 = Conv2D(filters * 2, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu',
                              name="CONV4_1X1_" + str(count))(embeddings)
        else:
            print("!!!ERROR!!!: Strides number is WRONG in Branch #4....")

        # Group all the layers
        concat_layer = Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
        final = Conv2D(filters * 2, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                       name="CONV_final_"+str(count))(concat_layer)
        return final


class TweetSentimentInceptionV2_3x3(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, filters=4, dense_units=64, dropout=0):
        print("---------------------------------------3x3---------------------------------------")
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # Reshape
        embeddings = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)
        final = self.get_inception_model(embeddings, filters, count=1, strides_level=2, n=3)

        # Flatten
        X = Flatten()(final)

        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        # Dropout
        X = Dropout(dropout, name="DROPOUT_1")(X)

        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_4")(X)

        # Final layer
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)

class TweetSentimentInceptionV2_5x5(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, filters=4, dense_units=64, dropout=0):
        print("---------------------------------------5x5---------------------------------------")
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # Reshape
        embeddings = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)
        final = self.get_inception_model(embeddings, filters, count=1, strides_level=2, n=5)

        # Flatten
        X = Flatten()(final)
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_4")(X)

        # Final layer
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)


class TweetSentimentInceptionV2_3x3_Multi(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, filters=4, dense_units=64, dropout=0):
        print("---------------------------------------MULTI 3x3---------------------------------------")
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # Reshape
        embeddings = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        layer1 = self.get_inception_model(embeddings, filters, count=1, strides_level=1, n=3)
        layer2 = self.get_inception_model(layer1, filters, count=2, strides_level=1, n=3)
        layer3 = self.get_inception_model(layer2, filters, count=3, strides_level=1, n=3)
        layer4 = self.get_inception_model(layer3, filters, count=4, strides_level=1, n=3)

        # # Group all the layers
        concat_layer = Concatenate(axis=-1)([layer1, layer2, layer3, layer4])
        final = Conv2D(filters*2, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                       name="CONV_final")(concat_layer)

        # Flatten
        X = Flatten()(final)
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_4")(X)

        # Final layer
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)


class TweetSentimentInceptionV2_5x5_Multi(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, filters=4, dense_units=64, dropout=0):
        print("---------------------------------------MULTI 5x5---------------------------------------")
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # Reshape
        embeddings = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        layer1 = self.get_inception_model(embeddings, filters, count=1, strides_level=1, n=5)
        layer2 = self.get_inception_model(layer1, filters, count=2, strides_level=1, n=5)
        layer3 = self.get_inception_model(layer2, filters, count=3, strides_level=1, n=5)
        layer4 = self.get_inception_model(layer3, filters, count=4, strides_level=1, n=5)

        # # Group all the layers
        concat_layer = Concatenate(axis=-1)([layer1, layer2, layer3, layer4])
        final = Conv2D(filters*2, kernel_size=(1,1), strides=(1, 1), padding='same', activation='relu',
                       name="CONV_final")(concat_layer)
        # Flatten
        X = Flatten()(final)
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_4")(X)

        # Final layer
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)
