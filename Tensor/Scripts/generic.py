import os
import pandas as pd


def load_dataframe(file_path, file_type):
    file_path = os.path.abspath('..') + "\\" + file_path

    if file_type == 'csv':
        return pd.read_csv(file_path, encoding='ISO-8859-1', engine='c', dtype=str)
    elif file_type == 'xlsx':
        return pd.read_excel(file_path, sheetname=0)
    elif file_type == 'txt':
        return pd.read_table(file_path, encoding='ISO-8859-1', engine='c', dtype=str)
    else:
        raise Exception("WRONG FILE TYPE!!")