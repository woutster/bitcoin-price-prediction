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

def get_bitcoin_data(limit):
	columns = ['Date', 'Polarity', 'Subjectivity']
	data = np.array(columns)
	for submission in reddit.subreddit('bitcoin').top(time_filter='year', limit=limit):
		dateStamp = UNIX_to_date(submission.created_utc)
		polarity, subjectivity = get_sentiment(submission.title)
		data = np.vstack((data, [dateStamp, polarity, subjectivity]))
	df = pd.DataFrame(data[1:, :], columns=['Date', 'Polarity', 'Subjectivity'])
	data = df.set_index('Date')
	return data

def process_data(save):
	processed_data = ['Date', 'NumberOfPosts', 'numberPositive',
					'numberNegative', 'TotalPos', 'TotalNeg',
					'AveragePolarity', 'AverageSubjectivity']
	data = get_bitcoin_data_2()
	for index, _ in data.iterrows():
		polarity = np.array(data.get_value(index, 'Polarity')).astype(np.float)
		subjectivity = np.array(data.get_value(index, 'Subjectivity')).astype(np.float)
		amount_pos = np.where(polarity > 0)[0].size
		amount_neg = np.where(polarity < 0)[0].size
		totalPos = polarity[np.where(polarity>0)].sum()
		totalNeg = polarity[np.where(polarity<0)].sum()
		processed_data = np.vstack((processed_data, [index, 
													polarity.size,
													amount_pos,
													amount_neg,
													totalPos,
													totalNeg,
													np.average(polarity), 
													np.average(subjectivity)]))
	df = pd.DataFrame(data=processed_data[1:,1:],
					index=processed_data[1:,0],
					columns=processed_data[0,1:])
	df = df[~df.index.duplicated(keep='first')]
	if save:
		df.to_csv(path_or_buf='reddit_api_features.csv', sep=',', header=True, index=True)
	else:
		return df

def get_bitcoin_data_2():
	columns = ['Date', 'Polarity', 'Subjectivity']
	data = np.array(columns)
	subreddit = reddit.subreddit('bitcoin')
	start_date = 1451606400 # 1 january 2016
	end_date = 1483228800 # 31 december 2016
	for submission in subreddit.submissions(start_date, end_date):
		dateStamp = UNIX_to_date(submission.created_utc)
		polarity, subjectivity = get_sentiment(submission.title)
		data = np.vstack((data, [dateStamp, polarity, subjectivity]))
	df = pd.DataFrame(data[1:, :], columns=['Date', 'Polarity', 'Subjectivity'])	
	return df.set_index('Date')