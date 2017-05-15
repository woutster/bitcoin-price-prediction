import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta, date
import datetime


printing = False

def daterange(start_date, end_date):
	for n in range(int ((end_date - start_date).days)):
		yield start_date + timedelta(n)


def test_data(printing):
	start_date = date(2016, 1, 1)
	end_date = date(2017, 1, 1)
	folder_name = 'good_data/'
	not_found_list, error_list, good_list = np.array(['date']), np.array(['date']), np.array(['date'])
	for single_date in daterange(start_date, end_date):
		my_file = Path(folder_name + single_date.strftime("%Y-%m-%d") + '_output.csv')
		if my_file.exists():
			try:
				data = pd.read_csv(my_file, sep=';', encoding='utf-8')
				good_list = np.vstack((good_list, data['date'].iloc[-1:]))
				if printing:
					print('Found!')
					print(data['date'].iloc[-1:])
					print()
			except:
				error_list = np.vstack((error_list, single_date.strftime("%Y-%m-%d") + '_output.csv'))
				if printing:
					print('~~~~~~~~~')
					print('Error!')
					print(single_date.strftime("%Y-%m-%d") + '_output.csv')
					print('~~~~~~~~~')
					print()
		else:
			not_found_list = np.vstack((not_found_list, single_date))
			if printing:
				print('Not found!')
				print(single_date.strftime("%Y-%m-%d") + '_output.csv')
				print()

	return good_list, error_list, not_found_list

def test_single_data(error_list):
	for file in error_list[1:,]:
		print(file[0])
		my_file = 'good_data/' + file[0]
		data = pd.read_csv(my_file, sep=';', encoding='utf-8')

