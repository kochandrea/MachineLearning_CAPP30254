import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import graphviz 
import seaborn as sns


def scatter_plot(df, x, y):
	'''
	Create scatter plot.

	Inputs:
		- df: dataframe
		- x: (str) attribute name
		- y: (str) attribute name
	'''
	plt.scatter(df[x], df[y])
	plt.title(y+" "+x)
	plt.xlabel(x) 
	plt.ylabel(y)
	plt.show()


def histogram_plot(df, attribute_name):
	'''
	Create histogram.

	Inputs:
		- df: dataframe
		- attribute_name (str)
	'''
	plt.hist(df[attribute_name], bins=20)
	plt.title('Distribution of '+attribute_name)
	plt.ylabel(attribute_name)
	plt.xlabel('Frequency')
	plt.show()


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


def find_most_funded(df, var_of_interest, groupby_var, top_n):
	'''
	Counts of top n most funded -- MAX n is 10
	'''
	return df.groupby(groupby_var)[var_of_interest].value_counts().nlargest(top_n)



def plot_pie_chart_top_n_funded(df, var_of_interest, groupby_var, top_n=""):
	if top_n != "":
		top_n_funded = find_most_funded(df, var_of_interest, groupby_var, top_n)
		sizes = list(top_n_funded)
		idx = list(top_n_funded.index)
		labels = []
		for tup in idx:
			labels.append(tup[1])

		possible_colors = ['#a8e6cf', '#dcedc1', '#ffd3b6', '#ffaaa5', 
							'#ff8b94', '#ffffba', '#a2798f',
							'#d7c6cf','#8caba8','#ebdada']

		colors = possible_colors[:top_n]
		explode = [0]*top_n 

		plt.pie(sizes, explode=explode, labels=labels, colors=colors,
			autopct='%1.1f%%', shadow=False, startangle=140)
		plt.title('Fully_funded by ' + str(var_of_interest) + ' : split of top '+ str(top_n))
		plt.axis('equal')
		plt.show()




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



