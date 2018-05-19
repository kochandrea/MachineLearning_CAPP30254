import numpy as np
import pandas as pd


def impute_mean(df, given_cols=[]):
	'''
	Replace NaN with mean of column

	ONLY RUN IF YOU ARE CERTAIN YOU WANT TO IMPUTE MEAN ON ALL COLUMNS

	Input:
		- df:  dataframe
	Output:
		- df:  cleaned dataframe
	'''
	if not given_cols == []:
		for col in given_cols:
			mean = df[col].mean()
			df[col].fillna(mean, inplace=True)
	else:
		missing = list(df.loc[:, df.isna().any()].columns)
		for col in missing:
			mean = df[col].mean()
			df[col].fillna(mean, inplace=True)
	return df


def drop_missing(df):
	'''
	Drop rows with columns containing NaN

	Input:
		- df:  dataframe
	Output:
		- df:  cleaned dataframe
	'''
	df = df.dropna()
	return df


def zscore_remove_outlier(df, attribute):
	'''
	Adapted from:  https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-pandas-dataframe
	'''
	df = df[((df[attribute] - df[attribute].mean()) / df[attribute].std()).abs() < 3]
	return df


def dummytize(df):
	for col in df.columns:
		if ((list(df[col].unique()) == ['f','t']) 
			or (list(df[col].unique()) == ['t','f'])):
			df[col] = df[col].apply(lambda x: 1 if x=='t' else 0)
	return df


def has_na(df):
	'''
	Identify columns which have missing data.
	
	Input:
		- df:  dataframe
	Output:
		- tup: (list) tuples of missing column name and count of NaN
	'''
	missing = list(df.loc[:, df.isna().any()].columns)
	count = []
	for col in missing:
		num = df[col].isnull().sum()
		count.append(num)
	tup = list(zip(missing, count))
	return tup


def discretize(df, column_name, num_bins, want_quantile):
	'''
	Discretizes a given column.

	Inputs:
		- df:  dataframe
		- column_name:  (str) column name for column to discretize
	Output:
		- df:  updated dataframe with column of discretized results
	'''
	new_col = column_name + ' - Discretized'
	if want_quantile:
		df[new_col] = pd.qcut(df[column_name], num_bins)
	else:
		df[new_col] = pd.cut(df[column_name], num_bins)
	return df