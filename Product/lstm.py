import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore") # Ignore warnings

# Gets a matrix as input and divides it into training and test sets
def load_data(matrix, seq_len, pred_len, pred_delay, normalise_window, ratio):
    sequence_length = seq_len + pred_len + pred_delay # The length of the slice that is taken from the data

    result = [] # List that is going to contain the sequences
    for index in range(len(matrix) - sequence_length): # Take every possible sequence from beginning to end
        result.append(matrix[index: index + sequence_length]) # Append sequence to result list

    result = np.array(result) # Convert result to numpy array

    row = round(ratio * result.shape[0]) # Up until this row the data is training data

    train = result[:int(row), :] # Get training data
    np.random.shuffle(train) # Random shuffle trainingdata
    x_train = train[:, :seq_len] # The sequence of the training data
    y_train = train[:, -pred_len] # The to be predicted values of the training data
    x_test = result[int(row):, :seq_len] # The sequence of the test data
    y_test = result[int(row):, -pred_len] # The to be predicted values of the test data

    if normalise_window: # Normalise
        mu = np.mean(matrix) # Mean
        sigma = np.std(matrix) # Deviation

        # y_train = (y_train - mu) / sigma
        # y_test = (y_test - mu) / sigma

        x_train = (x_train - mu) / sigma
        x_test = (x_test - mu) / sigma

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Reshape, because expected lstm_1_input to have 3 dimensions
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # Reshape, because expected lstm_1_input to have 3 dimensions

    return [x_train, y_train, x_test, y_test]


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of window_size steps before shifting prediction run forward by prediction_len steps

    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)): # -1
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    return prediction_seqs


# Plots the results
def plot_results_multiple(predicted_data, true_data, prediction_len, prediction_delay):
    true_data = true_data.reshape(len(true_data),1) # reshape true data from batches to one long sequence

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')

    if prediction_len == 1:
        plt.plot(predicted_data, label='Prediction')
        plt.legend()
    else:
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(predicted_data):
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()

    plt.show()

def plot_loss(model_fit):
    # summarize history for accuracy
    plt.plot(model_fit.history['loss'])
    plt.plot(model_fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# Compute mean error
def error(predicted, real, prediction_delay):
    errors = []
    predicted = np.array(predicted)
    predicted = predicted.reshape(np.size(predicted), 1)
    for p, r in zip(predicted, real):
        errors.append(np.abs(p-r))
    mean_error = np.array(errors).mean()
    return mean_error




def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted
