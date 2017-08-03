from ANNManager import Model
import DataHandler
import os
import generic
import pandas as pd

import json

settings = open("settings.json")
settings = json.load(settings)

print('Loading data...')

# Create a dict to store the accuracy of each secondary model
model_accuracy = {}

top_words = settings['TOP_WORDS']
ROOT_PATH = os.path.abspath('..')

batch_size = settings['BATCH_SIZE']
max_words_limit = settings['MAX_WORDS_LIMIT']

OUTPUT_PATH = ROOT_PATH + "\Data\\predictions.csv"
TEMP_PATH = ROOT_PATH + "\Data\\temp.csv"

CREATE_TRAIN_DATA = False
TRAIN_PRIMARY = True

# Check output file is closed
DataHandler.check_file_open(OUTPUT_PATH)

test_df = generic.load_dataframe("Data\\test.csv", "csv")
test_df.dropna(inplace=True)

test_data = DataHandler.get_testing_data(test_df, 'TEXT')

# ===============================================================================================
cluster_code = 0
model_id = "MODEL_" + str(cluster_code)
my_model = Model(X_train=[], Y_train=[], X_test=[], Y_test=[], top_words=top_words,
                      max_words_limit=max_words_limit,
                      num_classes=0, epochs=0, batch_size=batch_size, model_id=model_id)


my_model.revive()

predicted_classes, scores = my_model.predict(test_data, verbose=1)

# Reload LE
le_file_name = "LE_" + str(cluster_code)
le = DataHandler.load_pickle(ROOT_PATH + '\\Data\\PickleJar\\' + le_file_name + '.pkl')

# Get NMS from codes
predicted_classes = le.inverse_transform(predicted_classes)

# save data
test_df['PREDICTION'] = pd.Series(predicted_classes)
test_df['SCORE'] = pd.Series(scores)

test_df.to_csv(OUTPUT_PATH, index=False)
print('Results stored to disc @', OUTPUT_PATH)
