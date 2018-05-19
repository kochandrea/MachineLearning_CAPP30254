# Functions created for Homework 2.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import graphviz 
import seaborn as sns



def read_csv(filepath):
	'''
	Converts csv file into pandas dataframe.

	Input:
		- filepath
	Output:
		- dataframe
	'''
	df = pd.read_csv(filepath)
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


def review_df(df):
	'''
	Outputs shape information of dataframe.

	Input:
		- df:  dataframe
	'''
	num_rows, num_cols = df.shape
	total_obs = num_rows * num_cols
	print("Number of rows:  ", num_rows)
	print("Number of columns:  ", num_cols)
	print("Total observations:  ", total_obs)


def correlation_heatmap(df):
	'''
	Create correlation heatmap of data.

	Input:
		- df:  dataframe
	'''
	corr = df.corr()
	sns.heatmap(corr, 
				xticklabels=corr.columns.values,
				yticklabels=corr.columns.values)


def mean_Y_by_X(df, X, Y):
	'''
	Plots mean of Y by X.

	Input:
		- df:  dataframe
		- X: (str) independent variable name
		- Y: (str) dependent variable name
	'''
	mean_df = df.groupby(X).mean()
	x = list(mean_df.index.values)
	y = list(mean_df[Y].values)
	plt.plot(x, y)
	plt.ylabel(Y)
	plt.xlabel(X)
	plt.title('Mean '+Y+' by '+X)
	plt.show()


def replace_missing_with_mean(df):
	'''
	Replace NaN with mean of column

	Input:
		- df:  dataframe
	Output:
		- df:  cleaned dataframe
	'''
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


def discretize(df, column_name):
	'''
	Discretizes a given column.

	Inputs:
		- df:  dataframe
		- column_name:  (str) column name for column to discretize
	Output:
		- df:  updated dataframe with column of discretized results
	'''
	new_col = column_name + ' - Discretized'
	df[new_col] = pd.cut(df[column_name], 500)
	return df


def split(df, var, test_size):
	'''
	Spilts dataframe into train and test dataframes.

	Inputs:
		- df:  dataframe
		- var:  (str) dependent variable (a column name)
		- test_size:  (float) size (in percentage) of test data;
					  the split
	Outputs:
		- x_train, y_train, x_test, y_test:  (dataframes) the testing and
											 training dataframes
	'''
	X = df
	Y = df[var]
	x_train, x_test, y_train, y_test = train_test_split(X, Y, 
										test_size=test_size)
	return (x_train, x_test, y_train, y_test)


def classify_and_evaluate(x_train, x_test, y_train, y_test, n):
	'''
	Inputs:
		- x_train, y_train, x_test, y_test:  (dataframes) the testing and
											 training dataframes	
		- n:  (int) number of neighbors to get
	'''
	knn = KNeighborsClassifier(n_neighbors=n, metric='minkowski', 
								metric_params={'p': 3})
	knn.fit(x_train, y_train)
	print('Probability estimates for the test data X:  ')
	print(knn.predict_proba(x_test))
	print('Train score:  ')
	print(knn.score(x_train, y_train))
	print('Test score:  ')
	print(knn.score(x_test, y_test))


