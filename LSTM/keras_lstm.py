from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras import metrics
from keras.models import Sequential
from keras import callbacks
import keras.backend as K

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

import numpy as np
import random
from functools import partial
from itertools import product

import lstm_functions
import gather_data

# allow_soft_placement=True
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

# Filter compile warnings, errors still show
# export TF_CPP_MIN_LOG_LEVEL=2

###############################################################################
###############################  Parmaters  ###################################

label_threshold = 0.01
use_existing_data = True
single_data_array = True
shuffle = True
normalize = True # misschien andere manier
kfold = False
nfolds = 10

train_test_split = 0.9
num_classes = 3
seq_length = 3
data_dim = 1
random_seed = 985790549 #random.randint(0, 2**32-1)

lstm_size = 128 # 256 of 512 misschien beter
lstm_hidden_size = int((lstm_size+num_classes)/2)
batch_size = 64
no_epochs = 3000
learning_rate = 0.00003 # mischien iets hoger
drop_rate = 0.5
val_split = 0.1

###############################################################################
###########################  Building the model  ##############################

def make_model(seq_length, data_dim, lstm_size, drop_rate, lstm_hidden_size, num_classes):
    model = Sequential()
    model.add(LSTM(input_shape = (seq_length, data_dim), units = lstm_size))
    model.add(Dropout(drop_rate))
    model.add(Dense(lstm_hidden_size, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(units = num_classes, activation='softmax'))
    return model


def fit_model(model, train_X, train_y, batch_size, no_epochs, val_split, callbacks, verbose=True):
    return model.fit(
        train_X,
        train_y,
        batch_size = batch_size,
        epochs = no_epochs,
        validation_split = val_split,
        verbose=verbose,
        shuffle=True,
        callbacks = callbacks,
    )


###############################################################################
###########################  Training the model  ##############################


optimizer = optimizers.RMSprop(lr=learning_rate)
callbacks = [callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')]

X, y = gather_data.get_data(use_existing_data, single_data_array, label_threshold)
import pdb; pdb.set_trace()
X = X[:, data_dim]
y = y[:]


if kfold:
    train_X_kfold, train_y_kfold, test_X_kfold, test_y_kfold = lstm_functions.process_data(X,
        y, 
        normalize,
        shuffle,
        seq_length,
        random_seed,
        train_test_split,
        num_classes, 
        kfold,
        nfolds)

    total_accuracy = 0
    total_prec, total_rec, total_f1 = 0, 0, 0
    for i in range(0, nfolds):
        train_X = train_X_kfold[i]
        train_y = train_y_kfold[i]
        test_X = test_X_kfold[i]
        test_y = test_y_kfold[i]
        model = None # Clear model
        model = make_model(seq_length, data_dim, lstm_size, drop_rate, lstm_hidden_size, num_classes)
        model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
        print("Training %d fold out of %d..." % ((i+1), nfolds))
        model_fit = fit_model(model, train_X, train_y, batch_size, no_epochs, val_split, callbacks, verbose=0)
        acc, f1 = lstm_functions.get_statistics(model, 
            model_fit, 
            X, 
            train_X, 
            test_X, 
            train_y, 
            test_y, 
            seq_length,
            kfold,
            nfolds,
            showPlots=False)

        total_accuracy = total_accuracy + acc
        total_prec = total_prec + f1[0]
        total_rec = total_rec + f1[1]
        total_f1 = total_f1 + f1[2]

    total_accuracy = total_accuracy/nfolds
    total_prec = total_prec/nfolds
    total_rec = total_rec/nfolds
    total_f1 = total_f1/nfolds

    print("Average accuracy = %f" % total_accuracy)
    print("Average precission = %f" % total_prec)
    print("Average recall = %f" % total_rec)
    print("Average f1 score = %f" % total_f1)

else:
    train_X, train_y, test_X, test_y = lstm_functions.process_data(X, 
        y, 
        normalize,
        shuffle,
        seq_length,
        random_seed,
        train_test_split,
        num_classes, 
        kfold,
        nfolds)
    model = make_model(seq_length, data_dim, lstm_size, drop_rate, lstm_hidden_size, num_classes)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    model_fit = fit_model(model, train_X, train_y, batch_size, no_epochs, val_split, callbacks, verbose=2)
    lstm_functions.get_statistics(model, 
        model_fit, 
        X, 
        train_X, 
        test_X, 
        train_y, 
        test_y, 
        seq_length,
        kfold,
        nfolds,
        showPlots=True)
