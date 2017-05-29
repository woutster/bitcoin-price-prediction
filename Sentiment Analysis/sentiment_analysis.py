import re
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.stem import *
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
import numpy as np


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
    return text


def filter_stopword(text):
	stop = set(stopwords.words('english'))
	sentence = [i for i in text.split() if i not in stop]
	return ' '.join(sentence)


def return_sentiment(text):
	blob = TextBlob(text)
	polarity = 0
	subjectivity = 0
	for sentence in blob.sentences:
		polarity += sentence.sentiment.polarity
		subjectivity += sentence.sentiment.subjectivity
	return polarity, subjectivity


def test():
	sentiment_list1 = np.array(['sentence', 'polarity', 'subjectivity'])
	sentiment_list2 = np.array(['sentence', 'polarity', 'subjectivity'])
	filename = '../API/Twitter/good_data/2016-01-01_output.csv'
	data = pd.read_csv(filename, sep=';', encoding='utf-8')
	sentences = data['text']
	sentiment = data['sentiment']
	for line in data:
		text = preprocess_sentence(line)
		text = filter_stopword(text)
		polarity, subjectivity = return_sentiment(text)
		sentiment_list1 = np.vstack((sentiment_list1, [text, polarity, subjectivity]))
		import pdb; pdb.set_trace()

test()
