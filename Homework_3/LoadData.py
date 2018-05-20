import pandas as pd
import numpy as np 



def load_to_pd_df(filepath):
	'''
	Converts csv file into pandas dataframe.

	Input:
		- filepath
	Output:
		- dataframe
	'''
	if filepath[-3:] == 'csv':
		df = pd.read_csv(filepath)
	return df


def limit_date_range(df, date_range, date_variable):
	'''
	date_range = tuple
	Returns a pandas dataframe given date range.
	'''
	start_date, end_date = date_range
	df = df[(df[date_variable] >= start_date) & (df[date_variable] <= end_date)]
	return df


#Create dataframe of all variables, joining outcomes and projects on projectid
#Drop unwanted dates and set projectid as the index
date_range = ['2011-01-01', '2013-12-31']
outcomes_df = load_to_pd_df('kaggle_data/outcomes.csv')
projects_df = load_to_pd_df('kaggle_data/projects.csv')
projects_df = limit_date_range(projects_df, date_range, 'date_posted')
donors_choose_df = pd.merge(outcomes_df, projects_df, on='projectid')
donors_choose_df = donors_choose_df.set_index('projectid')


