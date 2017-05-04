# Get reddit data from /r/bitcoin

import praw
import numpy as np
import datetime
from textblob import TextBlob

reddit = praw.Reddit(client_id='oUo5NKou35H3iw',
                     client_secret='_ObY-dp3Li15IZLefdBVNJyQ_sA',
                     user_agent='btc_research')

def UNIX_to_date(timestamp):
	return datetime.datetime.fromtimestamp(
        int(timestamp)
    ).strftime('%Y-%d-%m')

def get_sentiment(string):
	blob = TextBlob(string)
	polarity = 0
	subjectivity = 0
	for sentence in blob.sentences:
		polarity += sentence.sentiment.polarity
		subjectivity += sentence.sentiment.subjectivity
	return polarity, subjectivity

def get_bitcoin_data():
	data = np.array(['Title', 'Date', 'Polarity', 'Subjectivity'])

	for submission in reddit.subreddit('bitcoin').hot(limit=10):
		dateStamp = UNIX_to_date(submission.created_utc)
		polarity, subjectivity = get_sentiment(submission.title)
		data = np.vstack((data, [submission.title, dateStamp, polarity, subjectivity]))

get_bitcoin_data()