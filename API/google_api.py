from pytrends.request import TrendReq
import pandas as pd

# Enter your own credentials
google_username = "bitcoin.predictor@gmail.com"
google_password = "..."

def get_google_data():
	pytrend1 = TrendReq(google_username, google_password, hl='en-US', tz=360, custom_useragent=None)
	pytrend2 = TrendReq(google_username, google_password, hl='en-US', tz=360, custom_useragent=None)
	pytrend3 = TrendReq(google_username, google_password, hl='en-US', tz=360, custom_useragent=None)
	pytrend4 = TrendReq(google_username, google_password, hl='en-US', tz=360, custom_useragent=None)

	pytrend1.build_payload(kw_list=['Bitcoin'], timeframe='2015-01-01 2015-06-01')
	pytrend2.build_payload(kw_list=['Bitcoin'], timeframe='2015-06-02 2015-12-31')
	pytrend3.build_payload(kw_list=['Bitcoin'], timeframe='2016-01-01 2016-06-01')
	pytrend4.build_payload(kw_list=['Bitcoin'], timeframe='2016-06-02 2016-12-31')

	data1 = pytrend1.interest_over_time().reset_index()
	data2 = pytrend2.interest_over_time().reset_index()
	data3 = pytrend3.interest_over_time().reset_index()
	data4 = pytrend4.interest_over_time().reset_index()
	data12 = data1.append(data2, ignore_index=True)
	data34 = data3.append(data4, ignore_index=True)
	return data12.append(data34, ignore_index=True)

	
def process_data(save):
	data = get_google_data()
	data.reset_index('date')

	if save:
		data.to_csv(path_or_buf='google_api_features_2_years.csv', sep=';', header=True, index=False)
	else:
		return

process_data(True)