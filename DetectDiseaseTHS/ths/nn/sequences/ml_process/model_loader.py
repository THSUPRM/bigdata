import csv
import numpy as np
from keras.models import model_from_json
from keras.optimizers import RMSprop
from ths.nn.metrics.multiple import accuracy, calculate_cm_metrics
from sklearn.metrics import confusion_matrix


class ModelLoader:
    def load_evaluate_model(json_route, h5_route, optimizer, learning_rate, x_test_route, y_test_route):
        json_file = open(json_route, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(h5_route)
        print("Loaded model from disk")

        params_compile = {}
        if optimizer == 'RMSPROP':
            rmsprop = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-06)
            params_compile['optimizer'] = rmsprop

        loaded_model.compile(loss="categorical_crossentropy", metrics=['accuracy'], **params_compile)
        x_test = []
        y_test = []

        with open(x_test_route, "r") as f:
            csv_file = csv.reader(f, delimiter=',')
            for r in csv_file:
                for i in range(0, len(r)):
                    element = []
                    for j in str(r[i]).split():
                        element.append(int(j))
                    x_test.append(np.array(element))

        with open(y_test_route, "r") as f:
            csv_file = csv.reader(f, delimiter=',')
            for r in csv_file:
                for i in range(0, len(r)):
                    y_test.append(int(r[i]))

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # print("x_test[0]: ", x_test[0])
        # print("TYPE(x_test[0]): ", type(x_test[0]))
        # print("TYPE(x_test): ", type(x_test))

        # print("y_test[0]: ", y_test[0])
        # print("TYPE(y_test[0]): ", type(y_test[0]))
        # print("TYPE(y_test): ", type(y_test))

        predicted = loaded_model.predict(x_test)
        predicted = np.argmax(predicted, axis=1)
        # print("predicted[0]: ", predicted[0])
        # print("TYPE(predicted[0]): ", type(predicted[0]))

        # Getting metrics
        acc = accuracy(y_test, predicted)
        c_matrix = confusion_matrix(y_test, predicted)
        prec_1, recall_1, f1_1, spec_1 = calculate_cm_metrics(c_matrix)

        print("Accuracy: ", acc)
        print("Precision 1: ", prec_1)
        print("Recall 1: ", recall_1)
        print("F1 Score 1: ", f1_1)
        print("Specificity 1: ", spec_1)