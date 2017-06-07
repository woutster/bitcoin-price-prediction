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
from functools import partial
from itertools import product

import lstm_functions
import gather_data

###############################################################################
###############################  Parmaters  ###################################

label_threshold = 0.01
use_existing_data = True
single_data_array = True
shuffle = True
normalize = True # misschien andere manier

train_test_split = 0.9
num_classes = 3
seq_length = 1
data_dim = 22
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

X, y = gather_data.get_data(use_existing_data, single_data_array, label_threshold)


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


X = X[:, :data_dim]
y = y[:]
train_X, train_y, test_X, test_y = lstm_functions.process_data(X, 
                                                            y, 
                                                            normalize,
                                                            shuffle,
                                                            seq_length,
                                                            random_seed,
                                                            train_test_split,
                                                            num_classes)

w_array = np.array(((1, 1, 1), (1, 1, 1), (1, 1, 1)))
w_array = {0: 1.1, 1: 1, 2:1.1}
ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'


model = Sequential()
model.add(LSTM(input_shape = (seq_length, data_dim), units = lstm_size)) # TODO: recurrent dropout
model.add(Dropout(drop_rate))
model.add(Dense(lstm_hidden_size, activation='relu'))
model.add(Dropout(drop_rate)) #TODO: test remove
model.add(Dense(units = num_classes, activation='softmax'))


###############################################################################
###########################  Training the model  ##############################


# optimizer = optimizers.Adam(lr=learning_rate)
optimizer = optimizers.RMSprop(lr=learning_rate)
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
    callbacks = callbacks,
    # class_weight = w_array
)

lstm_functions.get_statistics(model, 
                            model_fit, 
                            X, 
                            train_X, 
                            test_X, 
                            train_y, 
                            test_y, 
                            seq_length)