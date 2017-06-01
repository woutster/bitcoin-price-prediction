# Get all data from blockchain.info

import requests
import numpy as np
import pandas as pd
import datetime

def get_response(url):
	dictList = []
	data = requests.get(url).json()['values']
	df = pd.DataFrame(data)
	return df.set_index('x')

def UNIX_to_date(timestamp):
	return datetime.datetime.fromtimestamp(
        int(timestamp)
    ).strftime('%Y-%d-%m')

def interpolate(timestamp, array):
	timestamp_previous = timestamp - (24*60*60)
	timestamp_next = timestamp + (24*60*60)
	return (array.get_value(timestamp_previous, 'y') + array.get_value(timestamp_next, 'y'))/2


def usdToBtc(btc):
	url = "https://blockchain.info/tobtc?currency=USD&value=" + str(btc)
	return get_response(url)

def getMarketPriceBTCInUSD():
	url = "https://api.blockchain.info/charts/market-price?start=2015-01-01&ormat=json&timespan=731days"
	return get_response(url)

def getTradeVolume():
	url = "https://api.blockchain.info/charts/trade-volume?start=2015-01-01&ormat=json&timespan=731days"
	return get_response(url)

def getMedianConfirmationTime():
	url = "https://api.blockchain.info/charts/median-confirmation-time?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getBlocksSize():
	url = "https://api.blockchain.info/charts/blocks-size?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getCostPerTransactionPercent():
	url = "https://api.blockchain.info/charts/cost-per-transaction-percent?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getDifficulty():
	url = "https://api.blockchain.info/charts/miners-revenue?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getHashRate():
	url = "https://api.blockchain.info/charts/hash-rate?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getNumberOfTransactionsPerBlock():
	url = "https://api.blockchain.info/charts/n-transactions-per-block?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getTotalTransactions():
	url = "https://api.blockchain.info/charts/n-transactions-total?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getTotalAdressesTransactions():
	url = "https://api.blockchain.info/charts/n-unique-addresses?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getNumberOfCirculatingBitcoin():
	url = "https://api.blockchain.info/charts/total-bitcoins?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def getMarketCapetalization():
	url = "https://api.blockchain.info/charts/market-cap?start=2015-01-01&format=json&timespan=731days"
	return get_response(url)

def process_data(save):
	
	marketPriceBTCInUSD = getMarketPriceBTCInUSD()
	tradeVolume = getTradeVolume()
	medianConfirmationTime = getMedianConfirmationTime()
	blocksSize = getBlocksSize()
	costPerTransactionPercent = getCostPerTransactionPercent()
	difficulty = getDifficulty()
	hashRate = getHashRate()
	numberOfTransactionsPerBlock = getNumberOfTransactionsPerBlock()
	totalTransactions = getTotalTransactions()
	totalAdressesTransactions = getTotalAdressesTransactions()
	numberOfcirculatingBitcoins = getNumberOfCirculatingBitcoin()
	marketCapitalization = getMarketCapetalization()
	timestamps = marketPriceBTCInUSD.index
	# import pdb; pdb.set_trace()
	columns = ['marketPriceBTCInUSD', 'tradeVolume', 'medianConfirmationTime',
		'blocksSize', 'costPerTransactionPercent', 'difficulty', 'hashRate', 
		'numberOfTransactionsPerBlock', 'totalTransactions', 
		'totalAdressesTransactions', 'numberOfcirculatingBitcoins', 
		'marketCapitalization']

	df = pd.DataFrame(index=timestamps, columns=columns)
	data = np.append(['timestamp'], columns)
	for index, _ in df.iterrows():
		# for hour in range(24):
		dateTime = UNIX_to_date(index)
		# 	dateTime = datetime.datetime.strptime(dateTime, '%Y-%d-%m')
		# 	dateTime = (dateTime + datetime.timedelta(hours=hour)).strftime('%Y-%d-%m:%H')
		if not (index in tradeVolume.index.values):
			tradevolume = interpolate(index, tradeVolume)
		else:
			tradevolume = tradeVolume.get_value(index, 'y')

		data = np.vstack((data, [dateTime, 
			marketPriceBTCInUSD.get_value(index, 'y'), 
			tradevolume, 
			medianConfirmationTime.get_value(index, 'y'), 
			blocksSize.get_value(index, 'y'), 
			costPerTransactionPercent.get_value(index, 'y'), 
			difficulty.get_value(index, 'y'), 
			hashRate.get_value(index, 'y'), 
			numberOfTransactionsPerBlock.get_value(index, 'y'), 
			totalTransactions.get_value(index, 'y'), 
			totalAdressesTransactions.get_value(index, 'y'), 
			numberOfcirculatingBitcoins.get_value(index, 'y'), 
			marketCapitalization.get_value(index, 'y')]))
	df = pd.DataFrame(data=data[1:,1:],
							index=data[1:,0],
							columns=data[0,1:])
	if save:
		df.to_csv(path_or_buf='blockchain_api_features_2_years.csv', sep=';', header=True, index=True)
	else:
		return df

process_data(True)