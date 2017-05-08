import redditScraper
import blockchain
import google_api

import numpy as np
import pandas as pd

def get_data(use_existing_data):
	if use_existing_data:
		reddit_data = redditScraper.process_data(False).sort()
		google_data = google_api.process_data(False).sort()
		blockchain_data = blockchain.process_data(False).sort()
	else :
		column_list = ['google_data', 'date_2', 'marketPriceBTCInUSD',
		'tradeVolume', 'medianConfirmationTime', 'blocksSize',
		'costPerTransactionPercent', 'difficulty', 'hashRate', 
		'numberOfTransactionsPerBlock', 'totalTransactions',
		'totalAdressesTransactions', 'numberOfcirculatingBitcoins', 
		'marketCapitalization', 'date_3', 'NumberOfPosts', 'numberPositive',
		'numberNegative', 'TotalPos', 'TotalNeg', 'AveragePolarity', 
		'AverageSubjectivity']
		reddit_data = pd.read_csv(filepath_or_buffer='reddit_api_features.csv', sep=',', index_col=0).reset_index().values
		google_data = pd.read_csv(filepath_or_buffer='google_api_features.csv', sep=',', index_col=0).reset_index().values
		blockchain_data = pd.read_csv(filepath_or_buffer='blockchain_api_features.csv', sep=',', index_col=0).reset_index().values
		merge_list = np.column_stack((google_data, blockchain_data, np.flipud(reddit_data)))
		merged = pd.DataFrame(data=merge_list[0:,1:],
							index=merge_list[0:,0],
							columns=column_list).drop('date_2', 1).drop('date_3', 1)
		X = merged.drop('marketPriceBTCInUSD', 1)
		y = merged['marketPriceBTCInUSD']
		return X, y