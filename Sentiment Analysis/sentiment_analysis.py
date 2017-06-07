import re
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.stem import *
from nltk.corpus import stopwords
import numpy as np
from textblob.classifiers import NaiveBayesClassifier

def word_feats(words):
    return dict([(word, True) for word in words])

def preprocess_sentence(text):
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
    return filter_stopword(text)

def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

def filter_stopword(text):
    stop = set(stopwords.words('english'))
    sentence = [i for i in text.split() if i not in stop]
    return ' '.join(sentence)

def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        featureVector.append(w.lower())
    return featureVector

# def train_classifier(preProcess):

def return_sentiment_api(text, preproc):
    if preproc:
        text = preprocess_sentence(text)
    blob = TextBlob(text)
    polarity = 0
    subjectivity = 0
    for sentence in blob.sentences:
        polarity += sentence.sentiment.polarity
        subjectivity += sentence.sentiment.subjectivity
    return polarity, subjectivity

def test_trained_classifier(text, preproc, cl):
    if preproc:
        text = preprocess_sentence(text)
    return cl.classify(text)


def test():
    sentiment_list = np.array(['sentence', 
        'sentiment textblob no preprocessing',
        'sentiment textblob with preprocessing',
        'sentiment own classifier no preprocessing',
        'sentiment own classifier with preprocessing',
        'True label'
        ])
    cl = train_classifier(False)
    filename = 'testdata.manual.2009.06.14.csv'
    test_data = pd.read_csv(filename, sep=',', encoding='latin1', header=None)
    sentences = test_data[2]
    label = test_data[0]
    for line in sentences:
        import pdb; pdb.set_trace()
        trueLabel = int(test_data.loc[test_data[2] == line][0])
        textBlobNo, _ = return_sentiment_api(line, False)
        textBlobYes, _ = return_sentiment_api(line, True)
        ownNo = test_trained_classifier(line, False, cl)
        ownYes = test_trained_classifier(line, True, cl)
        sentiment_list = np.vstack((sentiment_list, [line, textBlobNo, 
            textBlobYes, ownNo, ownYes, trueLabel]))


# test()

df = pd.read_csv('training.1600000.processed.noemoticon.csv', sep=',', encoding='latin1', header=None)
# cols = [1, 0]
# df = df[cols]
# df[0] = df[0].map({0: 'neg', 4: 'pos'})
# negids = df.loc[df[0] == 'neg']
# posids = df.loc[df[0] == 'pos']
# print('Positive tweets length = %d' %len(posids))
# print('Negative tweets length = %d' %len(negids))
# train_data = pd.concat([negids, posids]).sample(frac=1)
# tuples = [tuple(x) for x in train_data.values]
# return NaiveBayesClassifier(tuples)
# Get tweet words
tweets = []
featureList = []
preProcess = False
print('Gathering features...')
for _, row in df.iterrows():
    sentiment = row[0]
    tweet = row[1]
    if preProcess:
        tweet = processTweet(tweet)
    featureVector = getFeatureVector(tweet)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
featureList = list(set(featureList))
print('Gathering training set...')
training_set = nltk.classify.util.apply_features(extract_features, tweets)
print('Training...')
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
testTweet = 'Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis'
print('Gathering test features...')
test_features = extract_features(getFeatureVector(testTweet))
print('Classifying test sentence...')
import pdb; pdb.set_trace()
print(NBClassifier.classify(test_features))