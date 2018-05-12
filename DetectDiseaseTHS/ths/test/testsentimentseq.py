import numpy as np
import csv
import math

from ths.utils.files import GloveEmbedding
from ths.utils.sentences import SentenceToIndices, PadSentences
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Embedding
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Function to create model, required for KerasClassifier
def create_model(G=None, max_len=0):
    ###### CREACION DEL MODELO ######
    # Embedding
    # create Keras embedding layer
    word_to_idx, idx_to_word, word_embeddings = G.read_embedding()
    # vocabulary_len = len(word_to_idx) + 1
    vocabulary_len = len(word_to_idx)
    emb_dimension = G.get_dimensions()
    # get the matrix for the sentences
    embedding_matrix = word_embeddings
    # embedding layer
    embedding_layer = Embedding(input_dim=vocabulary_len, output_dim=emb_dimension, input_length=max_len,
                                trainable=False, name="EMBEDDING")
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    first_layer_units = 128
    first_layer_dropout = 0.5
    second_layer_units = 128
    second_layer_dropout = 0.5
    relu_dense_layer = 64
    dense_layer_units = 3

    #Modelo
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(first_layer_units, return_sequences=True, name='LSTM_1', recurrent_dropout=0.4))
    model.add(Dropout(first_layer_dropout, name="DROPOUT_1"))
    model.add(Dense(200, activation='relu'))
    model.add(LSTM(second_layer_units, return_sequences=False, name="LSTM_2"))
    model.add(Dropout(second_layer_dropout, name="DROPOUT_2"))
    model.add(Dense(relu_dense_layer, activation='relu'))
    model.add(Dense(dense_layer_units))
    model.add(Activation("softmax", name="softmax_final"))
    model.compile(optimizer=RMSprop(decay=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
    return model
    ###### CREACION DEL MODELO ######

labeled_tweets_filename = "data/cleantextlabels3.csv"
embedding_filename = "data/glove.6B.50d.txt"
np.random.seed(11)
# open the file with tweets
X_all = []
Y_all = []
with open(labeled_tweets_filename, "r", encoding="ISO-8859-1") as f:
    i = 0
    csv_file = csv.reader(f, delimiter=',')
    for r in csv_file:
        if i != 0:
            tweet = r[0]
            label = r[1]
            X_all.append(tweet)
            Y_all.append(label)
        i = i + 1
print("Data Ingested")
num_data = len(X_all)
limit = math.ceil(num_data * 0.60)
X_train_sentences = X_all
Y_train = Y_all
G = GloveEmbedding(embedding_filename)
word_to_idx, idx_to_word, embedding = G.read_embedding()
S = SentenceToIndices(word_to_idx)
X_train_indices, max_len = S.map_sentence_list(X_train_sentences)
print("Train data mappend to indices")
P = PadSentences(max_len)
X_train_pad = P.pad_list(X_train_indices)
print("Train data padded")
# convert to numPY arrays
X_train = np.array(X_train_pad)
Y_train = np.array(Y_train)
Y_train = to_categorical(Y_train, num_classes=3)
print("Train data convert to numpy arrays")
model = KerasClassifier(build_fn=create_model(G, max_len))
print("Model created")
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
print(type(grid))
print("GRID SEARCH WORKING")
grid_result = grid.fit(X_train, Y_train)
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
