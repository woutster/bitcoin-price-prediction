import numpy as np
import pandas as pd
import glob
import datetime as dt
from textblob import TextBlob

def get_sentiment(string):
	blob = TextBlob(string)
	polarity = 0
	subjectivity = 0
	for sentence in blob.sentences:
		polarity += sentence.sentiment.polarity
		subjectivity += sentence.sentiment.subjectivity
	return polarity, subjectivity

def concatenate_data(save):
	dataframe = np.array(['id', 'date', 'retweets', 'favorites', 'text', 'geo', 'mentions', 'hashtags', 'id', 'permalink'])
	for filename in glob.iglob('good_data/*.csv'):
		data = pd.read_csv(filename, sep=';', encoding='utf-8')
		data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M').dt.strftime('%d-%m-%Y')
		dataframe = np.vstack((dataframe, data))
	df = pd.DataFrame(data=dataframe[1:,1:],
					  index=dataframe[1:,1],
					  columns=dataframe[0,1:])
	df.drop(df.columns[[0, 4, 5, 6, 7, 8]], axis=1, inplace=True)
	if save:
		df.to_csv(path_or_buf='all_data_twitter.csv', sep=';', header=True, index=True)
	else:
		return df

def preprocess_data():
	df = concatenate_data(False)
	for index, _ in df.iterrows():
		today = df.loc[index]
		polarity_list, subjectivity_list = ['polarity'], ['subjectivity']
		for text in today['text']:
			polarity, subjectivity = get_sentiment(text)
			polarity_list = np.vstack((polarity_list, polarity))
			subjectivity_list = np.vstack((subjectivity_list, subjectivity))
		polarity_list, subjectivity_list = polarity_list[1:], subjectivity_list[1:]
		today.loc[:, 'polarity'] = polarity_list
		today.loc[:, 'subjectivity'] = subjectivity_list
		import pdb; pdb.set_trace()
		


preprocess_data()	
