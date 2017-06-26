import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np


def preprocess_sentence(text, stopwords):
    # process the tweets

    #Convert to lower case
    text = text.lower()
    #Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(http://www\.\s[^\s]+)|(www\.\s[^\s]+)|(https?://[^\s]+)|(https?://\s[^\s]+))','URL', text)
    text = re.sub('\S*/\S*', '', text)
    #Convert @username to AT_USER
    text = re.sub('@[^\s]+','AT_USER', text)
    #Convert pic.twitter to PIC
    text = re.sub('pic.twitter.*', 'PIC', text)
    #Remove additional white spaces
    text = re.sub('[\s]+', ' ', text)
    #Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    #trim
    text = text.strip('\'"')
    return filter_stopword(text, stopwords)

def filter_stopword(text, stop):
    
    sentence = [i for i in text.split() if i not in stop]
    return ' '.join(sentence)

def return_sentiment_api(text, preproc, stopwords):
    if preproc:
        text = preprocess_sentence(text, stopwords)
    blob = TextBlob(text)
    polarity = 0
    subjectivity = 0
    for sentence in blob.sentences:
        polarity += sentence.sentiment.polarity
        subjectivity += sentence.sentiment.subjectivity
    return polarity, subjectivity


def scikit_model(df_train, df_test, preproc, stopwords):

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    

    print('Generating train data...')
    for _, row in df_train.iterrows():

        sentiment = row[0]
        tweet = row[1]
        if preproc:
            tweet = preprocess_sentence(tweet, stopwords)
        if sentiment == 0:
            train_labels.append('neg')
            train_data.append(tweet)
        elif sentiment == 4:
            train_labels.append('pos')
            train_data.append(tweet)

    print('Generating test data...')
    for _, row in df_test.iterrows():
        tweet = row[1]
        if preproc:
            tweet = preprocess_sentence(tweet, stopwords)
        sentiment = row[0]
        if sentiment == 0:
            test_data.append(tweet)
            test_labels.append('neg')
        elif sentiment == 4:
            test_data.append(tweet)
            test_labels.append('pos')

    print("Vectorizing data...")
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    print("Train classifier...")
    classifier_liblinear = svm.LinearSVC()
    clf = CalibratedClassifierCV(classifier_liblinear)
    t0 = time.time()
    clf.fit(train_vectors, train_labels)
    t1 = time.time()
    print("Classifying test data...")
    prediction_liblinear = clf.predict(test_vectors)
    t2 = time.time()
    prediction_liblinear = []
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1

    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))

def calculate_fscore(sentiment_list):
    sentiment_list = sentiment_list[1:]
    label_list = np.array(['labelNo', 'labelYes', 'TrueLabel'])
    for line in sentiment_list:
        scoreNo, scoreYes, trueScore = float(line[1]), float(line[2]), float(line[3])
        if scoreNo > 0:
            scoreNo = 4
        else:
            scoreNo = 0

        if scoreYes > 0:
            scoreYes = 4
        else:
            scoreYes = 0
        label_list = np.vstack((label_list, [scoreNo, scoreYes, trueScore]))

    precisionNo, recallNo, fscoreNo, supportNo = score(label_list[1:, 2], label_list[1:, 0])
    print('labels: {}'.format(['Negative', 'Positive']))
    print('precision: {}'.format(precisionNo))
    print('recall: {}'.format(recallNo))
    print('fscore: {}'.format(fscoreNo))
    print('support: {}'.format(supportNo))
    print('')
    precisionYes, recallYes, fscoreYes, supportYes = score(label_list[1:, 2], label_list[1:, 1])
    print('labels: {}'.format(['Negative', 'Positive']))
    print('precision: {}'.format(precisionYes))
    print('recall: {}'.format(recallYes))
    print('fscore: {}'.format(fscoreYes))
    print('support: {}'.format(supportYes))


def test():

    nltk_stopwords = set(stopwords.words('english'))
    sentiment_list = np.array(['sentence', 
        'sentiment textblob no preprocessing',
        'sentiment textblob with preprocessing',
        'True label'
        ])
    df_train = pd.read_csv('training.1600000.processed.noemoticon.csv', sep=',', encoding='latin1', header=None)
    df_test = pd.read_csv('testdata.manual.2009.06.14.csv', sep=',', encoding='latin1', header=None)

    print("Result scikit model without preprocessing: ")
    scikit_model(df_train, df_test, False, nltk_stopwords)
    print("###################")
    print("Result scikit model with preprocessing: ")
    scikit_model(df_train, df_test, True, nltk_stopwords)
    print("###################")

    sentences = df_test[1]
    label = df_test[0]
    for line in sentences:
        trueLabel = int(df_test.loc[df_test[1] == line][0])
        if trueLabel != 2:
            textBlobNo, _ = return_sentiment_api(line, False, nltk_stopwords)
            textBlobYes, _ = return_sentiment_api(line, True, nltk_stopwords) 
            sentiment_list = np.vstack((sentiment_list, [line, textBlobNo, 
                textBlobYes, trueLabel]))
    calculate_fscore(sentiment_list)

test()