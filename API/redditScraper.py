# Get reddit data from subreddits

import praw
import numpy as np
import datetime
import pandas as pd
from textblob import TextBlob
from datetime import timedelta, date

reddit = praw.Reddit(client_id='oUo5NKou35H3iw',
                     client_secret='_ObY-dp3Li15IZLefdBVNJyQ_sA',
                     user_agent='btc_research')

def UNIX_to_date(timestamp):
	return datetime.datetime.fromtimestamp(
        int(timestamp)
    ).strftime('%Y-%d-%m')

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_sentiment(string):
	blob = TextBlob(string)
	polarity = 0
	subjectivity = 0
	for sentence in blob.sentences:
		polarity += sentence.sentiment.polarity
		subjectivity += sentence.sentiment.subjectivity
	return polarity, subjectivity


def process_data(save):
	processed_data = ['Date', 'NumberOfPosts_bitcoin', 'NumberOfPosts_btc',
					'numberPositive_bitcoin', 'numberPositive_btc',
					'numberNegative_bitcoin', 'numberNegative_btc',
					'AverageScore_bitcoin', 'AverageScore_btc',
					'averageNoComments_bitcoin', 'averageNoComments_btc',
					'TotalPos_bitcoin', 'TotalPos_btc', 'TotalNeg_bitcoin',
					'TotalNeg_btc', 'AveragePolarity_bitcoin',
					'AveragePolarity_btc', 'AverageSubjectivity_bitcoin',
					'AverageSubjectivity_btc']
	data_bitcoin = get_bitcoin_data('bitcoin')
	data_btc = get_bitcoin_data('btc')
	start_date = date(2016, 1, 1)
	end_date = date(2017, 1, 1)
	for single_date in daterange(start_date, end_date):
		index = single_date.strftime("%Y-%d-%m")
		if index != "2017-01-01":
			polarity_bitcoin = np.array(data_bitcoin.get_value(index, 'Polarity')).astype(np.float)
			polarity_btc = np.array(data_btc.get_value(index, 'Polarity')).astype(np.float)
			score_bitcoin = np.array(data_bitcoin.get_value(index, 'Score')).astype(np.float)
			score_btc = np.array(data_btc.get_value(index, 'Score')).astype(np.float)
			num_comments_bitcoin = np.array(data_bitcoin.get_value(index, 'noOfComments')).astype(np.float)
			num_comments_btc = np.array(data_btc.get_value(index, 'noOfComments')).astype(np.float)
			subjectivity_bitcoin = np.array(data_bitcoin.get_value(index, 'Subjectivity')).astype(np.float)
			subjectivity_btc = np.array(data_btc.get_value(index, 'Subjectivity')).astype(np.float)
			amount_pos_bitcoin = np.where(polarity_bitcoin > 0)[0].size
			amount_neg_bitcoin = np.where(polarity_bitcoin < 0)[0].size
			amount_pos_btc = np.where(polarity_btc > 0)[0].size
			amount_neg_btc = np.where(polarity_btc < 0)[0].size
			totalPos_bitcoin = polarity_bitcoin[np.where(polarity_bitcoin>0)].sum()
			totalNeg_bitcoin = polarity_bitcoin[np.where(polarity_bitcoin<0)].sum()
			totalPos_btc = polarity_btc[np.where(polarity_btc>0)].sum()
			totalNeg_btc = polarity_btc[np.where(polarity_btc<0)].sum()
			processed_data = np.vstack((processed_data, [index, 
														polarity_bitcoin.size,
														polarity_btc.size,
														amount_pos_bitcoin,
														amount_pos_btc,
														amount_neg_bitcoin,
														amount_neg_btc,
														np.average(score_bitcoin),
														np.average(score_btc),
														np.average(num_comments_bitcoin),
														np.average(num_comments_btc),
														totalPos_bitcoin,
														totalPos_btc,
														totalNeg_bitcoin,
														totalNeg_btc,
														np.average(polarity_bitcoin),
														np.average(polarity_btc), 
														np.average(subjectivity_bitcoin),
													np.average(subjectivity_btc)]))

	df = pd.DataFrame(data=processed_data[1:,1:],
					index=processed_data[1:,0],
					columns=processed_data[0,1:])
	df = df[~df.index.duplicated(keep='first')]
	if save:
		df.to_csv(path_or_buf='reddit_api_features.csv', sep=';', header=True, index=True)
	else:
		return df

def get_bitcoin_data(subreddit):
	columns = ['Date', 'Polarity', 'Subjectivity', 'Score', 'noOfComments']
	data = np.array(columns)
	subreddit = reddit.subreddit(subreddit)
	start_date = 1451606400 # 1 january 2016
	end_date_test = 1452211200 # 8 january 2016
	end_date = 1483228800 # 1 january 2017
	for submission in subreddit.submissions(start_date, end_date):
		dateStamp = UNIX_to_date(submission.created_utc)
		polarity, subjectivity = get_sentiment(submission.title)
		data = np.vstack((data, [dateStamp, polarity, subjectivity, submission.score, submission.num_comments]))
	df = pd.DataFrame(data[1:, :], columns=columns)	
	return df.set_index('Date')