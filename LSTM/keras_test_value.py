from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.models import Sequential
from keras import callbacks
from sklearn.metrics import accuracy_score

import numpy as np
import warnings

import gather_data


warnings.filterwarnings("ignore")

label_threshold = 0.005
use_existing_data = True
single_data_array = True
train_test_split = 0.9
num_classes = 1
normalise = True
seq_length = 15
data_dim = 1

lstm_size = 128
batch_size = 32
no_epochs = 300
learning_rate = 0.001
drop_rate = 0.1
val_split = 0.2



def merge_data(a, b):   
    return np.c_[a, b]


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, dtype=float)
    return(data - mean) / std


def process_data(X, y, normalise):
    sequence_length = seq_length + 1

    if normalise:
        X = normalize_data(X)

    data = merge_data(X, y)
    data = data[:, :-1]

    result = []

    for index in range(data.shape[0] - sequence_length):
        result.append(data[index: index + sequence_length, :])

    result = np.array(result)

    split_index = int(train_test_split * result.shape[0])
    x_train = result[:split_index, :seq_length, :]
    x_test = result[split_index:, :seq_length, :]
        

    y_train = result[:split_index, 0, -1]
    y_test = result[split_index:, 0, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    return [x_train, y_train, x_test, y_test]

X, y = gather_data.get_data(use_existing_data, single_data_array, label_threshold)
X = X[:, 1:(1+data_dim)]
train_X, train_y, test_X, test_y = process_data(X, y, normalise)

model = Sequential()
model.add(LSTM(input_dim = data_dim, units = lstm_size))
model.add(Dense(output_dim = num_classes))
model.add(Dropout(drop_rate))
model.compile(loss = 'mse', optimizer = 'rmsprop') # compile

# callbacks = [callbacks.EarlyStopping(monitor='val_losss', min_delta=0, patience=0, mode='auto')]

model_fit = model.fit(
    train_X,
    train_y,
    batch_size = batch_size,
    epochs = no_epochs,
    validation_split = val_split,
    # callbacks = callbacks
)

def predict(model, data, window_size):
    prediction_seqs = []
    for i in range(len(data)): 
        input = np.reshape(data[i], (1, data[i].shape[0], data[1].shape[1]))
        prediction_seqs.append(np.argmax(model.predict(input)[0]))
    return prediction_seqs

# Compute mean error
def error(predicted, real):
    real = np.argmax(real, axis=1)
    error = accuracy_score(real, predicted)
    return error

def get_statistics():
    predictions_train = predict(model, train_X, seq_length)
    print("Real: ", train_y)
    print("Predicted: ", predictions_train)
    print("Error: ", error(np.array(predictions_train), train_y))

    predictions = predict(model, test_X, seq_length)
    print("Error: ", error(np.array(predictions), test_y))

# get_statistics()