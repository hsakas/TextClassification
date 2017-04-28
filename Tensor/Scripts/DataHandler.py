import pandas as pd
import os
import Vectorizer as vct
from tqdm import tqdm
import pickle
import numpy as np
import collections
from collections import OrderedDict, Counter
from keras.preprocessing.text import text_to_word_sequence
import math
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

import json

settings = open("settings.json")
settings = json.load(settings)

ROOT_PATH = os.path.abspath('..')


def pickle_dataframe(df, filepath):
    print('Pickling dataframe...')
    df.to_pickle(filepath)
    print('Pickled data @', filepath)


def load_pickle_dataframe(filepath):
    print('Un-Pickling from ', filepath)

    return pd.read_pickle(filepath)


def pickle_data(data, filepath):
    print('Pickling data @', filepath)
    filehandler = open(filepath, "wb")
    pickle.dump(data, filehandler)
    filehandler.close()


def load_pickle(filepath):
    print('Un-Pickling data @', filepath)
    file = open(filepath, 'rb')
    data = pickle.load(file)
    file.close()

    return data


def get_bow(list_of_strings, size=1000, min_words_len=3):

    print('Creating B.O.W.............')
    bow = vct.bag_of_words(list(set(list_of_strings)), remove_puncs=True, remove_digits=True, remove_alnums=True)

    # Remove smaller words
    bow_updated = {}
    for word, count in bow.items():
        if len(word) >= min_words_len:
            bow_updated[word] = count

    bow = dict(bow_updated)
    print('Found ', len(list(bow.keys())), 'words in BOW')
    bow = dict(collections.Counter(dict(bow)).most_common(size-3))
    bow = OrderedDict(sorted(bow.items()))

    print('Selected top', len(list(bow.keys()))+3, 'words...')

    bow['UNKNOWN_WORDS'] = 1
    return bow


def load_indexed_data(df, x_col_name, y_col_name, bow_size=100, pickle=False, bow=None, sample_count=150):

    df.dropna(subset=[x_col_name], axis=0, inplace=True)
    df = df.reset_index(drop=True)
    if bow is None:
        list_of_strings = df[x_col_name].tolist()
        bow = get_bow(list_of_strings, size=bow_size, min_words_len=settings['MINIMUM_WORDS_LENGTH'])

    if pickle:
        # Store bow
        pickle_data(bow, ROOT_PATH + '\\Data\\PickleJar\\bow.pkl')

    print('Formatting input data......')
    list_of_words = list(bow.keys())
    X = []
    Y = []

    # Sample X & Y to right balance
    X_values = df[x_col_name].tolist()
    Y_keys = df[y_col_name].tolist()

    X_data, Y_data = X_values, Y_keys

    for i in tqdm(range(len(X_data))):
        try:
            encoded_data = get_encoded_vector(list_of_words, str(X_data[i]))

            X.append(encoded_data)
            Y.append(int(Y_data[i]))
        except:
            pass

    if pickle:
        # Store data
        pickle_data(X, ROOT_PATH + '\\Data\\PickleJar\\X.pkl')
        pickle_data(Y, ROOT_PATH + '\\Data\\PickleJar\\Y.pkl')
    return np.array(X), np.array(Y)


def get_encoded_vector(list_of_words, new_string):

    porter = PorterStemmer()
    lmtz = WordNetLemmatizer()

    if 'START_SEQ' not in list_of_words:
        list_of_words.append('START_SEQ')

    if 'UNKNOWN_WORDS' not in list_of_words:
        list_of_words.append('UNKNOWN_WORDS')

    if 'END_SEQ' not in list_of_words:
        list_of_words.append('END_SEQ')

    # list_of_words += ['START_SEQ', 'UNKNOWN_WORDS', 'END_SEQ']
    tokens = text_to_word_sequence(new_string, lower=True, split=" ")

    # Stem and Lemmatize the data
    token_stemmed = []

    for token in tokens:
        try:
            token_stemmed.append(porter.stem(lmtz.lemmatize(token)))
        except:
            token_stemmed.append(token)

    tokens = list(token_stemmed)

    out = []

    all_unknown_words = True

    for token in tokens:
        if token in list_of_words:
            all_unknown_words = False
            out.append(list_of_words.index(token))
        else:
            out.append(list_of_words.index('UNKNOWN_WORDS'))
    if all_unknown_words:
        print('Sentence not recognised:', new_string)

    out = [list_of_words.index('START_SEQ')] + out + [list_of_words.index('END_SEQ')]
    return out


def get_testing_data(df, x_col_name):
    # load bow
    bow = load_pickle(ROOT_PATH + '\\Data\\PickleJar\\bow.pkl')
    list_of_words = list(bow.keys())

    X = []
    for i in range(df.shape[0]):
        X.append(get_encoded_vector(list_of_words, str(df[x_col_name][i])))

    return np.array(X)


def check_file_open(filepath):
    if os.path.isfile(filepath):
        try:
            myfile = open(filepath, "r+")
        except IOError:
            print("Output file is open, please close and try again")
            exit()


def sample_data(lower_margin, keys_list, values_list, remove_extra=True):
    keys_counter = Counter(keys_list)

    duplicates = []
    extra = []

    out_keys = []

    for key, count in keys_counter.items():
        if count < lower_margin:
            # Add duplicates

            difference = lower_margin - count
            values_present = values_list[keys_list.index(key):keys_list.index(key) + count]
            repetition = math.ceil(difference / len(values_present))

            # print('to get diff of', difference, 'array of size', len(values_present),
            # 'is repeated', repetition, 'times')

            addition = values_present * (repetition + 1)
            duplicates += addition[:difference]
            out_keys += [key] * difference
        elif remove_extra and (count > lower_margin):
            # Remove extra
            difference = count - lower_margin
            extra.extend([i for i, x in enumerate(keys_list) if x == key][:difference])

    # Get remaining values
    if remove_extra:
        values_list = np.delete(values_list, extra)
        keys_list = np.delete(keys_list, extra)
    return list(values_list) + duplicates, list(keys_list) + out_keys


def create_codes(df, column_name, revive=False, model_code=0):
    print('Encoding', column_name, '...')
    # get unique data
    nms_unique = df[column_name].unique().tolist()

    # fit model

    if not revive:
        print('Creating new Label Encoder...')
        le = LabelEncoder()
        le.fit(nms_unique)
    else:
        # Reload LE
        le_file_name = "LE_" + str(model_code)
        le = load_pickle(ROOT_PATH + '\\Data\\PickleJar\\' + le_file_name + '.pkl')
    # get all data
    nms = df[column_name].tolist()

    return le.transform(nms), le


def train_test_split(df, frac=0.8, random_state=200):
    print('Splitting dataset...')
    train = df.sample(frac=frac, random_state=random_state)
    test = df.drop(train.index)

    return train, test