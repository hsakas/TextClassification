"""
Library Name : Vectorizer
Author : Akash and Atiya
Licence : Freeware
"""

import numpy as np
from tqdm import tqdm
from collections import Counter
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import math
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

__all__ = ['count_vectorize', 'normalize', 'create_bag_of_words']
__version__ = '1.0.20'
__dependency__ = ['PatternHandler', 'tqdm', 'collections', 'numpy']
__author__ = "Akash & Atiya"


# Create instances
tfidf_vectorizer = TfidfVectorizer(min_df=1)


def custom_tokenizer(sentence, delimiters=['|', ','], remove_puncs=True, get_unique=False):
    # tokens = re.split('(\W)', sentence)
    for delimiter in delimiters:
        sentence = re.sub(re.escape(delimiter), " "+delimiter+" ", sentence)

    tokens = word_tokenize(sentence)

    # Remove duplicates
    if get_unique:
        tokens = list(set(tokens))

    if remove_puncs:
        tokens = [token for token in tokens if
                  not ((len(token.strip()) == 1) and bool(re.search("[^a-zA-Z0-9]", token)))]

    tokens = [token for token in tokens if (not bool(re.search("\s", token)) and token != '')]

    # Remove duplicates
    if get_unique:
        tokens = list(set(tokens))

    return tokens


def bag_of_words(list_of_strings, remove_puncs=True, remove_digits=True, remove_alnums=True):

    porter = PorterStemmer()
    lmtz = WordNetLemmatizer()

    # empty bag of words
    bag_of_words = []

    # Iterate for string
    for string in tqdm(list_of_strings):
        string_tokens = custom_tokenizer(string, remove_puncs=remove_puncs, get_unique=True)

        bag_of_words.extend(string_tokens)

    if remove_alnums:
        bag_of_words = [bag for bag in bag_of_words if bag.isalpha()]
    elif remove_digits:
        bag_of_words = [bag for bag in bag_of_words if (not isNumber(bag))]

    bag_of_words.sort()

    # Stem and Lemmatize the data
    bag_of_words_stemmed = []

    for word in bag_of_words:
        try:
            bag_of_words_stemmed.append(porter.stem(lmtz.lemmatize(word)))
        except:
            bag_of_words_stemmed.append(word)

    bag_of_words = list(bag_of_words_stemmed)

    # Remove stop words
    stop = set(stopwords.words('english'))
    print('Removing Stop words...')
    bag_of_words = [bag.strip().lower() for bag in bag_of_words if (bag.strip().lower() not in stop)]

    bow_counter = Counter(bag_of_words)
    bow_counter = OrderedDict(sorted(bow_counter.items()))

    return bow_counter


def isNumber(string):
    try:
        float(string)
        return True
    except:
        return False


