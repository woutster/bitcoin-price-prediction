import numpy as np
import pandas as pd
import glob
import datetime as dt
from textblob import TextBlob
from datetime import timedelta, date

def get_sentiment(string):
	blob = TextBlob(string)
	polarity = 0
	subjectivity = 0
	for sentence in blob.sentences:
		polarity += sentence.sentiment.polarity
		subjectivity += sentence.sentiment.subjectivity
	return polarity, subjectivity

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def concatenate_data(save):
	dataframe = np.array(['date', 'retweets', 'favorites', 'text'])
	for filename in glob.iglob('good_data/*.csv'):
		data = pd.read_csv(filename, sep=';', encoding='utf-8')
		data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M').dt.strftime('%d-%m-%Y')
		data['text'] = data['text'].map(str) + ' ' + data['geo'].map(str) + ' ' + data['mentions'].map(str) + ' ' + data['hashtags'].map(str)
		data.drop(data.columns[[0, 5, 6, 7, 8, 9]], axis=1, inplace=True)

		dataframe = np.vstack((dataframe, data))
	df = pd.DataFrame(data=dataframe[1:,1:],
					  index=dataframe[1:,0],
					  columns=dataframe[0,1:])
	if save:
		df.to_csv(path_or_buf='all_data_twitter.csv', sep=';', header=True, index=True)
		return df
	else:
		return df

def preprocess_data(save):
	df = concatenate_data(False)
	start_date = date(2016, 1, 1)
	end_date = date(2017, 1, 1)
	preprocessed_data = ['Date', 'numberOfPosts', 'numberOfPositive',
					'numberOfNegative', 'averagePolarity',
					'averageSubjectivity', 'averageRetweets',
					'averageFavourites']
	for single_date in daterange(start_date, end_date):
		index = single_date.strftime("%d-%m-%Y")
		today = df.loc[index]
		polarity_list, subjectivity_list = ['polarity'], ['subjectivity']
		for text in today['text']:
			polarity, subjectivity = get_sentiment(text)
			polarity_list = np.vstack((polarity_list, polarity))
			subjectivity_list = np.vstack((subjectivity_list, subjectivity))

		polarity_list, subjectivity_list = polarity_list[1:], subjectivity_list[1:]

		numberOfPosts = today.shape[0]
		numberOfPositive = np.where(polarity_list.astype(np.float) > 0)[0].size
		numberOfNegative = np.where(polarity_list.astype(np.float) < 0)[0].size
		avg_polarity = np.average(polarity_list.astype(np.float))
		avg_subjectivity = np.average(subjectivity_list.astype(np.float))
		averageRetweets = today['retweets'].mean()
		averageFavorites = today['favorites'].mean()

		preprocessed_data = np.vstack((preprocessed_data, [index, 
														numberOfPosts,
														numberOfPositive,
														numberOfNegative,
														avg_polarity,
														avg_subjectivity,
														averageRetweets,
														averageFavorites
														]))

	data = pd.DataFrame(data=preprocessed_data[1:,1:],
					index=preprocessed_data[1:,0],
					columns=preprocessed_data[0,1:])
	if save:
		data.to_csv(path_or_buf='twitter_features.csv', sep=';', header=True, index=True)
	else:
		return data


preprocess_data(True)	
