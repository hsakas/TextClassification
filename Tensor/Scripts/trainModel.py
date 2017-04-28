import json

settings = open("settings.json")
settings = json.load(settings)
# ---------------------- DATE TO REVIVE MODEL --------------------
REVIVE = input("Do you want to continue training previous model? (Y/N): ")

if REVIVE.lower().strip() == 'y':
    REVIVE = True
    print('Reviving the previous model...')
else:
    print('Training new model...')
    REVIVE = False


# --------------------- EPOCHS COUNT -------------------------------
print('________________________________________________________________')
print("Please enter number of Epochs")
EPOCHS_DAFAULT = settings['EPOCHS_DEFAULT']
EPOCHS = input("EPOCHS (Press enter for default value of " + str(EPOCHS_DAFAULT) + "): ")

if EPOCHS == '':
    EPOCHS = EPOCHS_DAFAULT
    print('Setting Epochs to default ' + str(EPOCHS_DAFAULT) + '..')

else:
    try:
        EPOCHS = int(EPOCHS)
    except:
        EPOCHS = EPOCHS_DAFAULT
        print('Invalid value entered!! Setting Epochs to default ' + str(EPOCHS_DAFAULT) + '..')

# ----------------------------- TRAIN MODEL -------------------------

import os
import numpy as np
import pandas as pd
import DataHandler
import generic
from ANNManager import Model

print('Loading data...')

# Create a dict to store the accuracy of each secondary model
model_accuracy = {}
model_code = 0
top_words = settings['TOP_WORDS']
ROOT_PATH = os.path.abspath('..')

# EPOCHS = 10
batch_size = settings['BATCH_SIZE']
max_words_limit = settings['MAX_WORDS_LIMIT']

OUTPUT_PATH = ROOT_PATH + "\Data\\predictions.csv"
CLUSTER_PATH = ROOT_PATH + "\Data\\cluster.csv"
TEMP_PATH = ROOT_PATH + "\Data\\temp.csv"

# Check output file is closed
DataHandler.check_file_open(OUTPUT_PATH)

# ---------------------------- create secondary models -----------------------------------------------

# assuming cluster zero
# save 0.json, 0.h5
# Load Clustered data

primary_df = generic.load_dataframe("Data\\train.csv", "csv")
primary_df.dropna(inplace=True)

evaluate_df = generic.load_dataframe("Data\\evaluate.csv", "csv")
evaluate_df.dropna(inplace=True)

# set all clusters to 0
primary_df['CLUSTER'] = 0
evaluate_df['CLUSTER'] = 0
# primary_df['CLASS'] = primary_df['NOUN'] + "|" + primary_df['MODIFIER']

# load bow
list_of_strings = primary_df['TEXT'].unique().tolist()

if not REVIVE:
    bow = DataHandler.get_bow(list_of_strings, size=top_words, min_words_len=3)
    DataHandler.pickle_data(bow, ROOT_PATH + '\\Data\\PickleJar\\bow.pkl')
else:
    # load bow
    bow = DataHandler.load_pickle(ROOT_PATH + '\\Data\\PickleJar\\bow.pkl')

# Create model id
model_id = 'MODEL_' + str(model_code)

# convert nm to codes and store in CLUSTER column
primary_df['CLASS_CODE'], le = pd.Series(DataHandler.create_codes(primary_df, 'CLASS', revive=REVIVE, model_code=model_code))
le_file_name = "LE_" + str(model_code)
DataHandler.pickle_data(le, ROOT_PATH + '\\Data\\PickleJar\\'+le_file_name+'.pkl')


# Get class codes for eveluate dataframe
evaluate_df['CLASS_CODE'], le = pd.Series(DataHandler.create_codes(evaluate_df, 'CLASS', revive=True, model_code=model_code))

# Get new X & Y for this cluster
X_data, Y_data = DataHandler.load_indexed_data(primary_df, 'TEXT', 'CLASS_CODE',
                                               bow_size=top_words, bow=bow)

X_evaluate, Y_evaluate = DataHandler.load_indexed_data(evaluate_df, 'TEXT', 'CLASS_CODE',
                                               bow_size=top_words, bow=bow)
X_train, Y_train = X_data, Y_data
# print('--------------X---------------------------')
# print(X_data)
# print('--------------Y--------------------------')
# print(Y_data)

num_classes = np.max(Y_train) + 1

# Train the model on new X, Y
secondary_model = Model(X_train, Y_train, X_evaluate, Y_evaluate, top_words, max_words_limit, num_classes,
                        EPOCHS, batch_size, model_id)

if not REVIVE:
    secondary_model.build()
    secondary_model.compile()
else:
    secondary_model.revive()

secondary_model.eval()
secondary_model.save()

model_accuracy[model_id] = secondary_model.accuracy

# pickle the data
DataHandler.pickle_data(model_accuracy, ROOT_PATH + '\\Data\\PickleJar\\accuracy.pkl')
