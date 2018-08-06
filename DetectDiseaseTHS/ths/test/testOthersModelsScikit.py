import keras.preprocessing.text as kpt
import numpy as np

from ths.nn.metrics import multiple as met
from keras.preprocessing.text import Tokenizer
from sklearn import svm
from sklearn.metrics import confusion_matrix

max_words = 10


def main():
    train_x = ['Trump is crazy', 'trump is bitching all the asdasda in live', 'Soccer is too slow',
              'Waste time in World Cup rum booze']
    train_y = np.array([1, 1, 0, 0])
    tokenizer = Tokenizer(num_words=max_words, oov_token='unk')
    # print(train_x)
    tokenizer.fit_on_texts(train_x)
    dictionary = tokenizer.word_index
    print("dictionary: ", dictionary)

    def convert_text_to_index_array(text):
        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        result =[]
        for word in kpt.text_to_word_sequence(text):
            print("word: ", word)
            x = dictionary.get(word, 0)
            print("x: ", x)
            result.append(x)
        return result
        #return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

    allWordIndices = tokenizer.texts_to_sequences(train_x)

    allWordIndices = np.asarray(allWordIndices)
    print("allWord 1: ", allWordIndices)
    train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    print("train_X", train_x)
    print("train_Y", train_y)

    print("type x: ", type(train_x))
    print("type y: ", type(train_y))

    print("SHAPE x: ", train_x.shape)
    print("SHAPE y: ", train_y.shape)

    print("train_X", train_x[0])
    print("train_Y", train_y[0])

    pred_tweet = ['Trump is live asdasda tu eres juan', 'Trump is asdasda illary', 'Trump is slow Soccer asdasda']
    # pred_tweet = ['Que mal estamos en Colombia', 'Estamos melos en Colombia', 'Sisas sisas en Colombia']
    pred_Y = np.array([0, 1, 1])

    allWordIndices = tokenizer.texts_to_sequences(pred_tweet)
    allWordIndices = np.asarray(allWordIndices)
    print("allWord 2: ", allWordIndices)
    pred_X = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    print("pred X: ", pred_X)


    #///////////////////////////////////////////////SVM///////////////////////////////////////////////#
    print("////////////////////////////////////Support Vector Machine////////////////////////////////////")
    clf = svm.SVC()
    clf.fit(train_x, train_y)

    # print("PREDICT: ", clf.predict(pred_X))

    predicted = clf.predict(pred_X)
    print("SVM Predicted: ", predicted)

    for x, y in zip(pred_Y, predicted):
        print("SVM expected value: " + str(x) + " predicted value: " + str(y))

    from sklearn.metrics import accuracy_score
    acc_scikit = accuracy_score(pred_Y, predicted)

    accuracy = met.accuracy(pred_Y, predicted)
    precision = met.precision_numpy(pred_Y, predicted)
    recall = met.recall_numpy(pred_Y, predicted)
    f1score = met.f1_numpy(pred_Y, predicted)
    fprate = met.fprate_numpy(pred_Y, predicted)

    print("SVM acc_scikit : ", acc_scikit)
    print("SVM ACC : ", accuracy)
    print("SVM PRECISION : ", precision)
    print("SVM RECALL : ",       recall)
    print("SVM F1SCORE : ",      f1score)
    print("SVM FPRATE :",        fprate)
    print("////////////////////////////////////Support Vector Machine////////////////////////////////////")
    #///////////////////////////////////////////////SVM///////////////////////////////////////////////#


    #///////////////////////////////////////////////NB///////////////////////////////////////////////#
    print("////////////////////////////////////Gaussian Naive Bayes////////////////////////////////////")
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(train_x, train_y)

    predicted = gnb.predict(pred_X)
    print("NB Predicted: ", predicted)

    for x, y in zip(pred_Y, predicted):
        print("NB expected value: " + str(x) + " predicted value: " + str(y))

    from sklearn.metrics import accuracy_score
    acc_scikit = accuracy_score(pred_Y, predicted)

    accuracy = met.accuracy(pred_Y, predicted)
    precision = met.precision_numpy(pred_Y, predicted)
    recall = met.recall_numpy(pred_Y, predicted)
    f1score = met.f1_numpy(pred_Y, predicted)
    fprate = met.fprate_numpy(pred_Y, predicted)

    print("NB acc_scikit : ", acc_scikit)
    print("NB ACC : ", accuracy)
    print("NB PRECISION : ", precision)
    print("NB RECALL : ", recall)
    print("NB F1SCORE : ", f1score)
    print("NB FPRATE :", fprate)
    print("////////////////////////////////////Gaussian Naive Bayes////////////////////////////////////")
    #////////////////////////////////////////////////NB///////////////////////////////////////////////#

    #//////////////////////////////////////////Logistic Regression//////////////////////////////////////////#
    print("////////////////////////////////Logistic Regression////////////////////////////////")
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(train_x, train_y)

    predicted = log_reg.predict(pred_X)

    for x, y in zip(pred_Y, predicted):
        print("Log_Reg expected value: " + str(x) + " predicted value: " + str(y))

    from sklearn.metrics import accuracy_score
    acc_scikit = accuracy_score(pred_Y, predicted)

    accuracy = met.accuracy(pred_Y, predicted)
    precision = met.precision_numpy(pred_Y, predicted)
    recall = met.recall_numpy(pred_Y, predicted)
    f1score = met.f1_numpy(pred_Y, predicted)
    fprate = met.fprate_numpy(pred_Y, predicted)

    print("Log_Reg acc_scikit : ", acc_scikit)
    print("Log_Reg ACC : ", accuracy)
    print("Log_Reg PRECISION : ", precision)
    print("Log_Reg RECALL : ", recall)
    print("Log_Reg F1SCORE : ", f1score)
    print("Log_Reg FPRATE :", fprate)
    print("////////////////////////////////////Logistic Regression////////////////////////////////////")
    #//////////////////////////////////////////Logistic Regression//////////////////////////////////////////#


if __name__ == "__main__":
    main()
