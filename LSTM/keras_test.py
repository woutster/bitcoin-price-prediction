from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.models import Sequential
from keras import callbacks
from keras import utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np
import gather_data
import matplotlib.pyplot as plt
import itertools
import random
import sys


# warnings.filterwarnings("ignore")

label_threshold = 0.01
use_existing_data = True
single_data_array = True
shuffle = True
normalise = True

train_test_split = 0.9
num_classes = 3
seq_length = 1
data_dim = 22
random_seed = 985790549 #random.randint(0, 2**32-1)

lstm_size = 128
lstm_hidden_size = int((lstm_size+num_classes)/2)
batch_size = 64
no_epochs = 3000
learning_rate = 0.00003
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
        print(random_seed)
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
model.add(Dense(lstm_hidden_size, activation='relu'))
model.add(Dropout(drop_rate))
model.add(Dense(units = num_classes, activation='softmax'))


optimizer = optimizers.Adam(lr=learning_rate)
# optimizer = optimizers.RMSprop(lr=learning_rate)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

callbacks = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')]

model_fit = model.fit(
    train_X,
    train_y,
    batch_size = batch_size,
    epochs = no_epochs,
    validation_split = val_split,
    verbose=True,
    shuffle=True,
    callbacks = callbacks
)

def predict(model, data, window_size):
    prediction = []
    probability = []
    for i in range(len(data)):
        input = np.reshape(data[i], (1, data[i].shape[0], data[1].shape[1]))
        prediction.append(np.argmax(model.predict(input)[0]))
        probability.append(model.predict(input)[0])
    return prediction, probability

# Compute mean error
def error(predicted, real):
    real = np.argmax(real, axis=1)
    error = accuracy_score(real, predicted)
    return error

def make_prediction_plot(real, predicted):
    axes = plt.gca()
    axes.set_ylim([-1, 3])
    plt.xlabel('Test set')
    plt.ylabel('Class')
    plt.title('Predicted class vs real class')
    plt.plot(real)
    plt.plot(predicted)
    plt.legend(['real classes', 'predicted classes'], loc='upper left')
    plt.show()

def make_loss_plot(loss, acc):
    plt.xlabel('Training epochs')
    plt.ylabel('Loss')
    plt.title('Training loss and accuracy on validation set')
    plt.plot(loss)
    plt.plot(acc)
    plt.legend(['validation loss', 'validation accuracy'], loc='upper left')
    plt.show()

def plot_confusion_matrix(cm, classes,title, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_statistics():
    predictions_train, probabilities_train = predict(model, train_X, seq_length)
    print("Percentage correct trainset: ", error(np.array(predictions_train), train_y) * 100, "%")

    predictions, probabilities = predict(model, test_X, seq_length)
    print("Real: ", np.argmax(test_y, axis=1))
    print("Predicted: ", predictions)
    print("Percentage correct testset: ", error(np.array(predictions), test_y) * 100, "%")
    make_prediction_plot(np.argmax(test_y, axis=1), predictions)
    make_loss_plot(model_fit.history.get('val_loss'), model_fit.history.get('val_acc'))
    cnf_matrix = confusion_matrix(np.argmax(test_y, axis=1), predictions)
    class_names = ['Fall', 'Stay', 'Rise']
    plt.figure()
    name = 'Confusion matrix of test data consisting of %d days' %len(predictions)
    plot_confusion_matrix(cnf_matrix, class_names, title=name)
    plt.show()

get_statistics()