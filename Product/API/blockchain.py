# Get all data from blockchain.info

import requests


def get_response(url):
	return requests.get(url).json()


def usdToBtc(btc):
	url = "https://blockchain.info/tobtc?currency=USD&value=" + str(btc)
	answer = get_response(url)

def getMarketPriceBTCInUSD():
	url = "https://api.blockchain.info/charts/market-price?start=2016-01-01&ormat=json"
	answer = get_response(url)

def getTradeVolume():
	url = "https://api.blockchain.info/charts/trade-volume?start=2016-01-01&ormat=json"
	answer = get_response(url)

def getMedianConfirmationTime():
	url = "https://api.blockchain.info/charts/median-confirmation-time?start=2016-01-01&format=json"
	answer = get_response(url):

def getBlocksSize():
	url = "https://api.blockchain.info/charts/blocks-size?start=2016-01-01&format=json"
	answer = get_response(url)

def getCostPerTransactionPercent():
	url = "https://api.blockchain.info/charts/cost-per-transaction-percent?start=2016-01-01&format=json"
	answer = get_response(url)

def getDifficulty():
	url = "https://api.blockchain.info/charts/miners-revenue?start=2016-01-01&format=json"
	answer = get_response(url)

def getHashRate():
	url = "https://api.blockchain.info/charts/hash-rate?start=2016-01-01&format=json"
	answer = get_response(url)

def getNumberOfTransactionsPerBlock():
	url = "https://api.blockchain.info/charts/n-transactions-per-block?start=2016-01-01&format=json"
	answer = get_response(url)

def getTotalTransactions():
	url = "https://api.blockchain.info/charts/n-transactions-total?start=2016-01-01&format=json"
	answer = get_response(url)

def getTotalAdressesTransactions():
	url = "https://api.blockchain.info/charts/n-unique-addresses?start=2016-01-01&format=json"
	answer = get_response(url)

def getNumberOfCirculatingBitcoin():
	url = "https://api.blockchain.info/charts/total-bitcoins?start=2016-01-01&format=json"
	answer = get_response(url)

def getTradeVolume():
	url = "https://api.blockchain.info/charts/trade-volume?start=2016-01-01&format=json"
	answer = get_response(url)

def getMarketCapetalization():
	url = "https://api.blockchain.info/charts/market-cap?start=2016-01-01&format=json"
	answer = get_response(url)
