# Get reddit data from /r/bitcoin

import praw
import numpy as np
import datetime
import pandas as pd
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
	columns = ['Date', 'Polarity', 'Subjectivity']
	data = np.array(columns)
	for submission in reddit.subreddit('bitcoin').hot(limit=50):
		dateStamp = UNIX_to_date(submission.created_utc)
		polarity, subjectivity = get_sentiment(submission.title)
		data = np.vstack((data, [dateStamp, polarity, subjectivity]))
	df = pd.DataFrame(data[1:, :], columns=['Date', 'Polarity', 'Subjectivity'])
	data = df.set_index('Date')
	return data

def process_data():
	data = get_bitcoin_data()
	for index, _ in data.iterrows():
		import pdb; pdb.set_trace()
		polarity = np.array(data.get_value(index, 'Polarity')).astype(np.float)
		subjectivity = np.array(data.get_value(index, 'Subjectivity')).astype(np.float)
		
	


process_data()