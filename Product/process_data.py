# Created by Wout Kooijman at 24-04-2017
# UvA BSc of Artificial Intelligence Thesis Code

# Data processing file


# Packages
import pandas as pd
import numpy as np
import pytrends
from pytrends.request import TrendReq

# API's
# from blockchain import statistics



def get_price_vectors():
    data = pd.read_csv('../Data/Prices/coinbase_data_2016.csv', na_values=['.'])

    data = data.drop('Time', 1).as_matrix()
    output_data = data[:, [0]]
    input_data = data[:, [1, 2]]

    return input_data, output_data

def get_google_data():
	# Connect to google
	pytrends = TrendReq('woutga@gmail.com', 'Arendpieter1', hl='en-US', tz=360, custom_useragent=None)

	# Get data
	kw_list = ['bitcoin']
	pytrends = pytrends.build_payload(kw_list, cat=0, timeframe='2016-01-01 2016-12-31', geo='', gprop='')
	print(pytrends)
	google_data = pytrends.interest_over_time()
	print(google_data)


def main():
	get_google_data()	

if __name__ == '__main__':
	main()


