# Instructions

## Load Project
1.	Open Project in PyCharm IDE & Set project interpreter as Anaconda

## Set-up Data
1.	To Divide the training data into train & evaluate files, Run Script “splitData.py”
a.	Input file: train_test.csv
b.	Output file: train.csv, evaluate.csv

   ### Variables to enter:
   Randomize the division of data: yes/no, if selected yes ‘y’, enter any number greater than 0
   Training data fraction: any fraction between 0-1 (e.g 0.8 divides the data into 80% training and 20% evaluation samples)

2.	To inflate data by creating duplicates, Run script “InflateAndSampleData.py”.
a.	Input file: train.csv
b.	Output file: train_samples.csv
c.	Change the train_sample.csv file format as per the train data template, add ID column and save file as train.csv

## Variables to set:
 
samples_count : Number of training samples per class
REMOVE_EXTRA: remove extra samples if samples count for any class is greater than the given samples_count number






## Model parameters
All the model parameters can be set/changed using the “settings.json” file:
 

1.	EPOCHS_DEFAULT: Default epochs count for training
2.	TOP_WORDS : Maximum number of words in Bag of words
3.	BATCH_SIZE: Training batch size
4.	MAX_WORDS_LIMIT: Maximum number of words in one text sample/answer
5.	MINIMUM_WORDS_LENGTH: Minimum length of a word to be added to Bag of words
6.	BASE_LR: Base learning rate
7.	OPTIMIZER: Training optimizer
8.	EMBEDDING_VECTOR_LENGTH: Length of embedding vector
9.	CNN_NO_OF_FILTER: Number of filter in CNN
10.	CNN_FILTER_LENGTH: filter length in CNN
11.	CNN_POOL_LENGTH: Pooling size for max pooling
12.	LSTM_CELLS_COUNT: Number of LSTM cells
13.	DROPOUT: Drop out in the model








## Train Model
Run Script “trainModel.py”
a.	Input file: train.csv
b.	Output file: Model & Data Pickles to be used to predictions (Model & PickleJar folder)

## Once the trainModel file is executed user is prompted to enter following options:
1.	Train new Model or Continue training previously trained Model:
To continue training using the previously trained Model, enter “y” in the console. To train new model enter “n”
2.	Number of Epochs
Enter an integer number to set epochs for training the model.
Leave blank to select default value from the settings.json file.

More classes can be added for training by adding more classes in the “CLASS” column of train.csv file. And the run trainModel script to train the Model with the updated classes structure.

## Test/Predict
Run Script “predictModel.py”
a.	Input file: test.csv, 
b.	Default inputs: Trained Model & Pickled data (Model & PickleJar folder)
c.	Output file: predictions.csv 

