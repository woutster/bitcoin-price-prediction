import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd
from nltk.corpus import stopwords



df_train = pd.read_csv('training.1600000.processed.noemoticon.csv', sep=',', encoding='latin1', header=None)
df_test = pd.read_csv('testdata.manual.2009.06.14.csv', sep=',', encoding='latin1', header=None)

preprocess_sentence = False

train_data = []
train_labels = []
test_data = []
test_labels = []
stopwords = set(stopwords.words('english'))

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


print('Generating train data...')
for _, row in df_train.iterrows():
    sentiment = row[0]
    tweet = row[1]
    if preprocess_sentence:
        tweet = preprocess_sentence(tweet, stopwords)
    if sentiment == 0:
        train_labels.append('neg')
        train_data.append(tweet)
    elif sentiment == 4:
        train_labels.append('pos')
        train_data.append(tweet)

print('Generating test data...')
for _, row in df_test.iterrows():
    tweet = row[2]
    if preprocess_sentence:
        tweet = preprocess_sentence(row[2], stopwords)
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
t0 = time.time()
classifier_liblinear.fit(train_vectors, train_labels)
t1 = time.time()
print("Classifying test data...")
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(test_labels, prediction_liblinear))