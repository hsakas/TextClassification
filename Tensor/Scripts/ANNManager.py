import json
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.optimizers import SGD, Adam
import keras
import numpy as np
import os
import pandas as pd

settings = open("settings.json")
settings = json.load(settings)

ROOT_PATH = os.path.abspath('..')

OPTIMIZER = settings['OPTIMIZER']
BASE_LR = settings['BASE_LR']

if OPTIMIZER.lower() == 'sgd':
    print('Optimizer selected:', OPTIMIZER)
    OPTIMIZER = SGD(lr=BASE_LR)
else:
    print('Optimizer selected:', OPTIMIZER)
    OPTIMIZER = Adam(lr=BASE_LR)


class Model:

    def __init__(self, X_train, Y_train, X_test, Y_test, top_words, max_words_limit, num_classes,
                 epochs, batch_size, model_id, optimizer=OPTIMIZER):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.top_words = top_words
        self.max_words_limit = max_words_limit
        self.num_classes = num_classes
        self.EPOCHS = epochs
        self.batch_size = batch_size
        self.model = None
        self.model_id = model_id
        self.accuracy = 0.0
        self.optimizer = optimizer

    def build(self):
        print('\nBuilding model...')
        # create the model
        embedding_vector_length = settings['EMBEDDING_VECTOR_LENGTH']
        self.model = Sequential()
        # self.model.add(Dropout(0.2, input_shape=(self.max_words_limit,)))
        self.model.add(Embedding(self.top_words, embedding_vector_length, input_length=self.max_words_limit))
        self.model.add(Convolution1D(nb_filter=settings['CNN_NO_OF_FILTER'], filter_length=settings['CNN_FILTER_LENGTH'], border_mode='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_length=settings['CNN_POOL_LENGTH']))
        self.model.add(LSTM(settings['LSTM_CELLS_COUNT']))
        self.model.add(Dropout(settings['DROPOUT']))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        print(self.model.summary())

    def eval(self):
        print(len(self.X_train), 'train sequences')
        print(self.num_classes, 'classes')

        # 'Convert class vector to binary class matrix '
        #      '(for use with categorical_crossentropy)')
        Y_train = keras.utils.to_categorical(self.Y_train, self.num_classes)
        Y_test = keras.utils.to_categorical(self.Y_test, self.num_classes)
        print('y_train shape:', Y_train.shape)
        print('y_test shape:', Y_test.shape)

        # fix random seed for reproducibility
        numpy.random.seed(7)
        # truncate and pad input sequences
        # max_words_limit = 500
        X_train = sequence.pad_sequences(self.X_train, maxlen=self.max_words_limit)
        X_test = sequence.pad_sequences(self.X_test, maxlen=self.max_words_limit)

        for epoch in range(self.EPOCHS):
            print('Epoch: '+str(epoch+1)+'/'+str(self.EPOCHS))

            self.model.fit(X_train, Y_train, nb_epoch=1, batch_size=self.batch_size)
            # Final evaluation of the model
            scores = self.model.evaluate(X_test, Y_test, verbose=0)

            prev_accuracy = self.accuracy
            self.accuracy = (scores[1] * 100)

            # Check if higher accuracy model found
            if self.accuracy > prev_accuracy:
                self.save()

            print("\nAccuracy: %.2f%%" % self.accuracy)

            # Saving stats...
            try:
                stats_df = pd.DataFrame({'Epoch': [epoch+1], 'Accuracy': [self.accuracy],
                                         'TOTAL EPOCHS': [self.EPOCHS]})
                stats_df.to_csv(ROOT_PATH + "\\Data\\stats.csv", index=False)

                print('Saved stats to file', ROOT_PATH + "\\Data\\stats.csv")
            except:
                print("Couldn't save stats...")
        # self.model.fit(X_train, Y_train, nb_epoch=self.EPOCHS, batch_size=self.batch_size)
        # # Final evaluation of the model
        # scores = self.model.evaluate(X_test, Y_test, verbose=0)
        # self.accuracy = (scores[1] * 100)
        # print("\nAccuracy: %.2f%%" % (scores[1] * 100))

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def predict(self, test_data, verbose):
        print(len(test_data), 'test sequences')

        # truncate and pad input sequences
        test_data = sequence.pad_sequences(test_data, maxlen=self.max_words_limit)

        # Predict data
        results = self.model.predict(test_data, self.batch_size, verbose=verbose)
        predicted_classes, scores = self.get_prediction(results)

        return predicted_classes, scores

    def save(self):
        # serialize model to json
        model_json = self.model.to_json()

        # Set paths
        model_path = ROOT_PATH + "\Data\Models\\" + self.model_id + ".json"
        weights_path = ROOT_PATH + "\Data\Models\\" + self.model_id + ".h5"

        with open(model_path, "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(weights_path)
        print('\nModel saved to disk', model_path)

    def revive(self):
        # set paths
        model_path = ROOT_PATH + "\Data\Models\\" + self.model_id + ".json"
        weights_path = ROOT_PATH + "\Data\Models\\" + self.model_id + ".h5"

        # load model from json
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_path)
        self.model = loaded_model
        self.compile()


    @staticmethod
    def get_prediction(predictions):
        scores = []
        predicted_classes = []
        for prediction in predictions:
            score = np.max(prediction)
            predicted_class = list(prediction).index(score)
            scores.append(score)
            predicted_classes.append(predicted_class)
        return predicted_classes, scores
