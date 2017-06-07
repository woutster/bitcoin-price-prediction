import itertools
from keras import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def make_one_hot(array, num_classes):
    array_one_hot = utils.to_categorical(array, num_classes)
    return array_one_hot


def merge_data(a, b):
    return np.c_[a, b]


def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None


def normalize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, dtype=float)
    return(data - mean) / std


def process_data(X, y, normalize, shuffle, seq_length, random_seed, train_test_split, num_classes):
    sequence_length = seq_length + 1

    if normalize:
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

    return [x_train, make_one_hot(y_train, num_classes), x_test, make_one_hot(y_test, num_classes)]


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


def make_plot(x, xlabel, ylabel, title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x)
    plt.show()


def make_plot_2(x1, x2, xlabel, ylabel1, ylabel2, title):
    fig, ax1 = plt.subplots()
    ax1.plot(x1, 'b')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color='b')
    ax1.tick_params('y', colors='b')


    ax2 = ax1.twinx()
    ax2.plot(x2, 'r')
    ax2.set_ylabel(ylabel2, color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
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

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_data_plots(X):
    price = X[:, 0]
    google = X[:, 12]
    noposts = X[:, 13]
    nopos = X[:, 14]
    noneg = X[:, 15]
    x_axis = 'days after 1 jan 2014'
    y_axis = 'btc price'
    make_plot_2(price, google, x_axis, y_axis, 'Google query amount', 'google query amount vs bitcoin price')
    make_plot_2(price, noposts, x_axis, y_axis, 'Number of bitcoin related posts on reddit', 'No. of bitcoin posts on reddit vs bitcoin price')
    make_plot_2(price, nopos, x_axis, y_axis, 'Number of positive posts on reddit', 'No. of positive bitcoin posts on reddit vs bitcoin price')
    make_plot_2(price, noneg, x_axis, y_axis, 'Number of negative posts on reddit', 'No. of negative bitcoin posts on reddit vs bitcoin price')


def get_statistics(model, model_fit, X, train_X, test_X, train_y, test_y, seq_length):
    predictions_train, probabilities_train = predict(model, train_X, seq_length)
    
    print("Percentage correct trainset: ", error(np.array(predictions_train), train_y) * 100, "%")

    predictions, probabilities = predict(model, test_X, seq_length)

    print("Real: ", np.argmax(test_y, axis=1))
    print("Predicted: ", predictions)
    print("Percentage correct testset: ", error(np.array(predictions), test_y) * 100, "%")

    make_prediction_plot(np.argmax(test_y, axis=1), predictions)
    loss = model_fit.history.get('val_loss')
    make_plot(loss, 'Training epochs', 'Loss', 'Training loss on validation set')
    acc = model_fit.history.get('val_acc')
    make_plot(acc, 'Accuracy', 'Training epochs', 'Training accuracy on validation set')
    prob = np.max(probabilities_train, axis=1) * 100
    make_plot(prob, 'Training epochs', 'certainty in percentages', 'Certainty data is labeled correctly')

    cnf_matrix = confusion_matrix(np.argmax(test_y, axis=1), predictions)
    class_names = ['Fall', 'Stay', 'Rise']
    plt.figure()
    name = 'Confusion matrix of test data consisting of %d days' %len(predictions)
    plot_confusion_matrix(cnf_matrix, class_names, title=name)
    plt.show()

    get_data_plots(X)