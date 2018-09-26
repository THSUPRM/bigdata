import numpy as np

np.random.seed(7)

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, AveragePooling2D, \
    Concatenate, ZeroPadding2D

from keras.layers.embeddings import Embedding


class  TweetSentiment2DCNN2Channel:
    def __init__(self, max_sentence_len, embedding_builder):
        self.max_sentence_len = max_sentence_len
        self.embedding_builder = embedding_builder
        self.model = None

    def build(self, first_dropout=0.0, padding='same', filters=4, kernel_size=(1,1), strides=(1,1), activation='relu',
              dense_units=64, second_dropout=0.0):

        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        embeddings2 = embeddings_layer(reverse_sentence_input)

        # Reshape
        embeddings1= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        embeddings2= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings2)

        #stack both input to make it a 2 chanell input
        concat_embeddings = Concatenate(axis = -1)([embeddings1, embeddings2])
        print("concat_embeddings: ", concat_embeddings)
        # Reshape with channels
        #X = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        # First convolutional layer
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        X = Conv2D(filters=20, kernel_size=kernel_size, strides=(1, 1), padding=padding, activation=activation,
                   name="CONV2D_1")(concat_embeddings)
        #X  = Conv2D(filters = 66, kernel_size = (kernel_height+2, 1),  strides=(1, 1), padding='same', activation=activation,
        #           name="CONV2D_2")(X)
        #MAX pooling
        pool_height =  self.max_sentence_len - kernel_height + 1  # assumes zero padding and stride of 1
        pool_size = (pool_height, 1)
        X = AveragePooling2D(pool_size=pool_size, name = "MAXPOOL_1")(X)

        #Flatten
        X = Flatten()(X)

        # Attention
        #att_dense = 70*20*1
        #attention_probs = Dense(att_dense, activation='softmax', name='attention_probs')(X)
        #attention_mul = Multiply(name='attention_multiply')([X, attention_probs])


        # # First dense layer
        dense_units = 128
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_1")(X)
        X = Dropout(second_dropout, name="DROPOUT_1")(X)

        # # Second dense layer
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dropout(second_dropout, name="DROPOUT_2")(X)
        #
        # # Third layer
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dropout(second_dropout, name="DROPOUT_3")(X)

        # Final layer
        X = Dense(1, activation= "sigmoid", name="FINAL_SIGMOID")(X)
        # create the model
        self.model = Model(input=[sentence_input, reverse_sentence_input] , output=X)

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

    #def sentiment_string(self, sentiment):
    #    return self.sentiment_map[sentiment]

    def save_model(self, json_filename, h5_filename):
        json_model = self.model.to_json()
        with open(json_filename, "w") as json_file:
            json_file.write(json_model)
        self.model.save_weights(h5_filename)
        return


class TweetSentimentInceptionOneChan(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1), activation='relu', dense_units=64,
              dropout=0):
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        # Reshape
        embeddings1= Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        #compute 1x1 convolution on input
        onebyone = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding=padding, activation=activation,
                   name="CONV_1X1_1")(embeddings1)
        #compute 3xdimension convolution on one by one
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        threebydim1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding,
                             activation=activation, name="CONV_3xdim_1")(onebyone)
        #compute 3xdimension convolution on input
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 3
        kernel_size = (kernel_height, kernel_width)
        threebydim2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding,
                             activation=activation, name="CONV_3xdim_2")(embeddings1)
        #compute 5xdimension convolution on one by one
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 5
        kernel_size = (kernel_height, kernel_width)
        fivebydim1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding,
                            activation=activation, name="CONV_5xdim_1")(onebyone)
        fivebydim1 = ZeroPadding2D((1, 0))(fivebydim1)
        #compute 5xdimension convolution on input
        kernel_width = self.embedding_builder.get_dimensions()
        kernel_height = 5
        kernel_size = (kernel_height, kernel_width)
        fivebydim2 = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding=padding,
                            activation=activation, name="CONV_5xdim_2")(embeddings1)
        fivebydim2 = ZeroPadding2D((1,0))(fivebydim2)
        # Group all the layers
        concat_layer = Concatenate(axis=-1)([threebydim1, threebydim2, fivebydim1, fivebydim2])
        final_onebyone = Conv2D(filters=filters, kernel_size=(1,1), strides=(1, 1), padding=padding,
                                activation=activation, name="CONV_1X1_final")(concat_layer)
        # Flatten
        X = Flatten()(final_onebyone)

        # try:
        #     if dropout != 0:
        #         X = Dropout(dropout, name="Dropout")(X)
        # except Exception as e:
        #     print(e)
        #     print("ERROR creating Dropout layer in the CNN model")
        #X = Dropout(0.10, name="DROPOUT_1")(X)

        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_4")(X)

        # Final layer
        #X = Dense(1, activation="sigmoid", name="FINAL_SIGMOID")(X)
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)


class TweetSentimentInceptionV2_3x3(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1), activation='relu', dense_units=64,
              dropout=0):
        print("---------------------------------------3x3---------------------------------------")
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        # Reshape
        embeddings1 = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        # Branch No. 1
        branch_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_1XN")(embeddings1)
        branch_1 = Conv2D(filters, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_1X3")(branch_1)
        branch_1 = Conv2D(filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_3X1")(branch_1)
        branch_1 = Conv2D(filters*2, kernel_size=(1, 3), strides=(2, 2), padding='same', activation='relu',
                          name="CONV1_2_1X3")(branch_1)
        branch_1 = Conv2D(filters*2, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_2_3X1")(branch_1)
        # Branch No. 2
        branch_2 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV2_1X1")(embeddings1)
        branch_2 = Conv2D(filters, kernel_size=(1, 3), strides=(2, 2), padding='same', activation='relu',
                          name="CONV2_1X3")(branch_2)
        branch_2 = Conv2D(filters, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV2_3X1")(branch_2)
        # Branch No. 3
        branch_3 = MaxPooling2D((1, 1), strides=(2, 2), padding='same', name='MAXPOL_1X1')(embeddings1)
        branch_3 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV3_1X1")(branch_3)
        # Branch No. 4
        branch_4 = Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu',
                          name="CONV4_1X1")(embeddings1)

        # # Group all the layers
        concat_layer = Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
        final = Conv2D(filters*2, kernel_size=(1,1), strides=(1, 1), padding='same',
                                activation='relu', name="CONV_final")(concat_layer)

        # Flatten
        X = Flatten()(final)

        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_4")(X)

        # Final layer
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)

class TweetSentimentInceptionV2_5x5(TweetSentiment2DCNN2Channel):
    def __init__(self, max_sentence_len, embedding_builder):
            super().__init__(max_sentence_len, embedding_builder)

    def build(self, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1), activation='relu', dense_units=64,
              dropout=0):
        print("---------------------------------------5x5---------------------------------------")
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings1 = embeddings_layer(sentence_input)
        # Reshape
        embeddings1 = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings1)
        # Branch No. 1
        branch_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_1XN")(embeddings1)
        branch_1 = Conv2D(filters, kernel_size=(1, 5), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_1X3")(branch_1)
        branch_1 = Conv2D(filters, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_3X1")(branch_1)
        branch_1 = Conv2D(filters*2, kernel_size=(1, 5), strides=(2, 2), padding='same', activation='relu',
                          name="CONV1_2_1X3")(branch_1)
        branch_1 = Conv2D(filters*2, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV1_2_3X1")(branch_1)
        # Branch No. 2
        branch_2 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV2_1X1")(embeddings1)
        branch_2 = Conv2D(filters, kernel_size=(1, 5), strides=(2, 2), padding='same', activation='relu',
                          name="CONV2_1X3")(branch_2)
        branch_2 = Conv2D(filters, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV2_3X1")(branch_2)
        # Branch No. 3
        branch_3 = MaxPooling2D((1, 1), strides=(2, 2), padding='same', name='MAXPOL_1X1')(embeddings1)
        branch_3 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                          name="CONV3_1X1")(branch_3)
        # Branch No. 4
        branch_4 = Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu',
                          name="CONV4_1X1")(embeddings1)

        # # Group all the layers
        concat_layer = Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
        final = Conv2D(filters*2, kernel_size=(1,1), strides=(1, 1), padding='same',
                                activation='relu', name="CONV_final")(concat_layer)

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

    def get_inception_model(self, embeddings, filters):
         # Branch No. 1
         branch_1 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                           name="CONV1_1XN")(embeddings)
         branch_1 = Conv2D(filters, kernel_size=(1, 5), strides=(1, 1), padding='same', activation='relu',
                           name="CONV1_1X3")(branch_1)
         branch_1 = Conv2D(filters, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu',
                           name="CONV1_3X1")(branch_1)
         branch_1 = Conv2D(filters * 2, kernel_size=(1, 5), strides=(2, 2), padding='same', activation='relu',
                           name="CONV1_2_1X3")(branch_1)
         branch_1 = Conv2D(filters * 2, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu',
                           name="CONV1_2_3X1")(branch_1)
         # Branch No. 2
         branch_2 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                           name="CONV2_1X1")(embeddings)
         branch_2 = Conv2D(filters, kernel_size=(1, 5), strides=(2, 2), padding='same', activation='relu',
                           name="CONV2_1X3")(branch_2)
         branch_2 = Conv2D(filters, kernel_size=(5, 1), strides=(1, 1), padding='same', activation='relu',
                           name="CONV2_3X1")(branch_2)
         # Branch No. 3
         branch_3 = MaxPooling2D((1, 1), strides=(2, 2), padding='same', name='MAXPOL_1X1')(embeddings)
         branch_3 = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
                           name="CONV3_1X1")(branch_3)
         # Branch No. 4
         branch_4 = Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu',
                           name="CONV4_1X1")(embeddings)

         # # Group all the layers
         concat_layer = Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
         final = Conv2D(filters * 2, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        activation='relu', name="CONV_final")(concat_layer)
         return final

    def build(self, padding='same', filters=4, kernel_size=(1, 1), strides=(1, 1), activation='relu', dense_units=64,
              dropout=0):
        print("---------------------------------------MULTI 5x5---------------------------------------")
        # Input Layer 1 - tweet in right order
        sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_1")
        #reverse_sentence_input = Input(shape=(self.max_sentence_len,), name="INPUT_2")
        # Embedding layer
        embeddings_layer = self.pretrained_embedding_layer()
        embeddings = embeddings_layer(sentence_input)
        # Reshape
        embeddings = Reshape((self.max_sentence_len, self.embedding_builder.get_dimensions(), 1))(embeddings)

        layer1 = self.get_inception_model(embeddings, filters)
        layer2 = self.get_inception_model(layer1, filters)
        # layer3 = self.get_inception_model(layer2, filters)
        # layer4 = self.get_inception_model(layer3, filters)

        # # Group all the layers
        concat_layer = Concatenate(axis=-1)([layer1, layer2])
        final = Conv2D(filters*2, kernel_size=(1,1), strides=(1, 1), padding='same',
                                activation='relu', name="CONV_final")(concat_layer)

        # Flatten
        X = Flatten()(final)
        X = Dense(units=dense_units, activation='relu', name="DENSE_2")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_3")(X)
        X = Dense(units=int(dense_units/2), activation='relu', name="DENSE_4")(X)

        # Final layer
        X = Dense(3, activation="softmax", name="FINAL_SOFTMAX")(X)
        # create the model
        self.model = Model(input=[sentence_input], output=X)
