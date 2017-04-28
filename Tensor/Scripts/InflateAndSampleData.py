import generic
import math
import DataHandler as dh
import pandas as pd
import os
from tqdm import tqdm

samples_count = 500
REMOVE_EXTRA = False

FILE_PATH = "\\Data\\train.csv"
OUT_PATH = os.path.abspath("..") + "\\Data\\train_sampled.csv"

df = generic.load_dataframe(FILE_PATH, "csv")
df.fillna('', inplace=True)


# Now remove extra data
x = df['TEXT'].tolist()
y = df['CLASS'].tolist()

print('Sampling data...')
x, y = dh.sample_data(lower_margin=samples_count, keys_list=list(y), values_list=list(x), remove_extra=REMOVE_EXTRA)
out_df = pd.DataFrame({'TEXT': x, "CLASS": y})

print('_______________________________________________________________')
print('Class Samples count:')
print('_______________________________________________________________')
print(out_df['CLASS'].value_counts(sort=True, ascending=False))

print('Saving output...')
out_df.to_csv(OUT_PATH, index=False)

print('Output saved @', OUT_PATH)


