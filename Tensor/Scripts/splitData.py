import os
import DataHandler
import generic

ROOT_PATH = os.path.abspath('..')

FILE_PATH = "\Data\\train_test.csv"
TRAIN_PATH = ROOT_PATH + "\Data\\train.csv"
TEST_PATH = ROOT_PATH + "\Data\\evaluate.csv"
random_state = 200
df = generic.load_dataframe(FILE_PATH, "csv")
frac = float(input("Enter training data fraction: "))
change_random = input("Change random state? Y/N: ")

if change_random.upper() == 'Y':
    random_state = int(input("Enter new random state: "))
elif change_random.upper() != 'N':
    print('Invalid value entered!! Proceeding with default state of', random_state)
train, test = DataHandler.train_test_split(df, frac=frac, random_state=random_state)

train.dropna(inplace=True)
test.dropna(inplace=True)

train.to_csv(TRAIN_PATH, index=False)
test.to_csv(TEST_PATH, index=False)
