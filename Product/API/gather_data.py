import redditScraper
import blockchain
import google_api

import pandas as pd

get_data(use_existing_data):
	if use_existing_data:
		reddit_data = redditScraper.process_data(False)
		blockchain_data = blockchain.process_data(False)
		google_data = google_api.process_data(False)
	else :
		reddit_data = pd.read_csv(filepath_or_buffer='reddit_api_features.csv', sep=',')
		blockchain_data = pd.read_csv(filepath_or_buffer='blockchain_api_features.csv', sep=',')
		google_data = pd.read_csv(filepath_or_buffer='google_api_features.csv', sep=',')
		import pdb; pdb.set_trace()

get_data(False)
