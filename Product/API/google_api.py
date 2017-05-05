from pytrends.request import TrendReq
import pandas as pd

# Enter your own credentials
google_username = "bitcoin.predictor@gmail.com"
google_password = "..."

def get_google_data():
	pytrend1 = TrendReq(google_username, google_password, hl='en-US', tz=360, custom_useragent=None)
	pytrend2 = TrendReq(google_username, google_password, hl='en-US', tz=360, custom_useragent=None)

	pytrend1.build_payload(kw_list=['Bitcoin'], timeframe='2016-01-01 2016-06-01')
	pytrend2.build_payload(kw_list=['Bitcoin'], timeframe='2016-06-02 2016-12-31')

	data1 = pytrend1.interest_over_time().reset_index()
	data2 = pytrend2.interest_over_time().reset_index()
	return data1.append(data2, ignore_index=True)

	
def process_data(save):
	data = get_google_data()
	data.reset_index('date')

	if save:
		data.to_csv(path_or_buf='google_api_features.csv', sep=',', header=True, index=False)
	else:
		return data