from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.models import Sequential
from keras import callbacks
from keras import utils
from sklearn.metrics import accuracy_score

import numpy as np
import warnings

import gather_data


# warnings.filterwarnings("ignore")

label_threshold = 0.005
use_existing_data = True
single_data_array = True
shuffle = True
normalise = True

train_test_split = 0.9
num_classes = 3
seq_length = 15
data_dim = 3
random_seed = 1

lstm_size = 128
batch_size = 64
no_epochs = 300
learning_rate = 0.0001
drop_rate = 0.5
val_split = 0.1


def make_one_hot(array):
    array_one_hot = utils.to_categorical(array, num_classes)
    return array_one_hot


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


    result = []
    for index in range(data.shape[0] - sequence_length):
        result.append(data[index: index + sequence_length, :])

    result = np.array(result)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(result)

    split_index = int(train_test_split * result.shape[0])
    x_train = result[:split_index, :seq_length, :-1]
    x_test = result[split_index:, :seq_length, :-1]
        

    y_train = result[:split_index, 0, -1]
    y_test = result[split_index:, 0, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    return [x_train, make_one_hot(y_train), x_test, make_one_hot(y_test)]

X, y = gather_data.get_data(use_existing_data, single_data_array, label_threshold)
X = X[:, :data_dim]
train_X, train_y, test_X, test_y = process_data(X, y, normalise)
model = Sequential()
model.add(LSTM(input_shape = (seq_length, data_dim), units = lstm_size)) # statefull? activation='sigmoid', return_sequences=False
model.add(Dropout(drop_rate))
model.add(Dense(10, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(units = num_classes, activation='softmax'))


# optimizer = optimizers.Adam(lr=learning_rate)
optimizer = optimizers.RMSprop(lr=learning_rate)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer) # loss = categorical_crossentropy ?

# callbacks = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, mode='auto')]

model_fit = model.fit(
    train_X,
    train_y,
    batch_size = batch_size,
    epochs = no_epochs,
    validation_split = val_split,
    verbose=True,
    shuffle=True
    #   callbacks = callbacks
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
    print("Percentage correct trainset: ", error(np.array(predictions_train), train_y) * 100, "%")

    predictions = predict(model, test_X, seq_length)
    print("Real: ", np.argmax(test_y, axis=1))
    print("Predicted: ", predictions)
    print("Percentage correct testset: ", error(np.array(predictions), test_y) * 100, "%")

get_statistics()