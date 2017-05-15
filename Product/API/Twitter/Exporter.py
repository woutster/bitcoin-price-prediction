# -*- coding: utf-8 -*-

import sys,getopt,got,datetime,codecs
from datetime import timedelta, date
import test_twitter_data

def daterange(start_date, end_date):
	for n in range(int ((end_date - start_date).days)):
		yield start_date + timedelta(n)


def export(query, since, until, output_file):

	only_popular = True
	threshold_popular = 5
	output_file = output_file + "_output.csv"
	outputFile = codecs.open(output_file, "w+", "utf-8")
 
	try:
		tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query).setSince(since).setUntil(until)

		outputFile.write('username;date;retweets;favorites;text;geo;mentions;hashtags;id;permalink')
		
		print('Searching...\n')
		
		def receiveBuffer(tweets):
			for t in tweets:
				if only_popular:
					if t.retweets > threshold_popular:
						outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;"%s";%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
				else:
					outputFile.write(('\n%s;%s;%d;%d;"%s";%s;%s;%s;"%s";%s' % (t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink)))
					
			outputFile.flush();
			print('Time = ' + tweets[0].date.strftime("%Y-%m-%d %H:%M"))
		
		got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)
		
	except:
		print('error, something went wrong')
	finally:
		outputFile.close()
		print('Done. Output file generated "' + output_file + '".')

def get_data(sequential):
	start_date = date(2016, 7, 8)
	end_date = date(2016, 7, 9)
	if sequential:
		for single_date in daterange(start_date, end_date):
			since = single_date.strftime("%Y-%m-%d")
			until = (single_date + timedelta(days=1)).strftime("%Y-%m-%d")
			query = 'bitcoin'
			print('Batch: ' + since)
			export(query, since, until, since)
	else:
		_, _, not_found_list = test_twitter_data.test_data(False)
		half = not_found_list.size
		for single_date in not_found_list[1:half]:
			single_date = single_date[0]
			since = single_date.strftime("%Y-%m-%d")
			until = (single_date + timedelta(days=1)).strftime("%Y-%m-%d")
			query = 'bitcoin'
			print('Batch: ' + since)
			export(query, since, until, since)

if __name__ == '__main__':
	get_data(False)
	