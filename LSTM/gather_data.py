import numpy as np
import pandas as pd


def get_labels(data, label_threshold):
	label_list = [0]
	for i in range(1, len(data)):
		change_percent = ((data[i]-data[i-1])/data[i-1])*100
		if change_percent > (label_threshold*100):
			label_list.append(2)
		elif change_percent < (-label_threshold*100):
			label_list.append(0)
		else:
			label_list.append(1)
	return np.array(label_list)

def get_data(use_existing_data, single_data_array, label_threshold):
	if use_existing_data:
		column_list = ['marketPriceBTCInUSD',
			'tradeVolume', 'medianConfirmationTime', 'blocksSize',
			'costPerTransactionPercent', 'difficulty', 'hashRate',
			'numberOfTransactionsPerBlock', 'totalTransactions',
			'totalAdressesTransactions', 'numberOfcirculatingBitcoins',
			'marketCapitalization', 'date_2', 'google_data', 'date_3', 
			'NumberOfPosts_bitcoin', 'numberPositive_bitcoin',
			'numberNegative_bitcoin', 'AverageScore_bitcoin',
			'averageNoComments_bitcoin', 'TotalPos_bitcoin',
			'TotalNeg_bitcoin', 'AveragePolarity_bitcoin',
			'AverageSubjectivity_bitcoin']

		blockchain_data = pd.read_csv(filepath_or_buffer=r'../API/blockchain_api_features_3_years.csv', sep=';', index_col=0).reset_index().values
		google_data = pd.read_csv(filepath_or_buffer=r'../API/google_api_features_3_years.csv', sep=';', index_col=0).reset_index().values
		reddit_data = pd.read_csv(filepath_or_buffer=r'../API/reddit_api_features_bitcoin_3_years.csv', sep=';', index_col=0).reset_index().values
		
		
		merge_list = np.column_stack((blockchain_data, google_data, np.flipud(reddit_data)))
		merged = pd.DataFrame(data=merge_list[0:,1:],
							index=merge_list[0:,0],
							columns=column_list).drop('date_2', 1).drop('date_3', 1)
		X = np.array(merged)
		y = get_labels(merged['marketPriceBTCInUSD'], label_threshold)
		return X, y
	# TODO
	else :
		print('Error, this is not yet implemented')
		# import redditScraper
		# import blockchain
		# import google_api
		# reddit_data = redditScraper.process_data(False).sort()
		# google_data = google_api.process_data(False).sort()
		# blockchain_data = blockchain.process_data(False).sort()