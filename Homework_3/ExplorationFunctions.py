import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import graphviz 
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


def basic_pie_chart(df, column): 
    vals = list(df.groupby(column)['fully_funded'].value_counts())
    keys = list(df.groupby(column)['fully_funded'].value_counts().keys())
    patches, texts = plt.pie(vals, startangle=90)
    plt.legend(patches, keys, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.title("Percent Funded(1) and Not Funded(0) by {}".format(column))
    plt.show()


def donut_plot(df, column):
    vals = list(df.groupby(column)['fully_funded'].value_counts())
    keys = list(df.groupby(column)['fully_funded'].value_counts().keys())
    fig = {"data": [{"values": vals,
                     "labels": keys,
                     "domain": {"x": [0, .48]},
                     "hoverinfo":"label+percent",
                     "hole": .4,
                     "type": "pie"}],
           "layout": {"title":"Percent Funded(1) and Not Funded(0) by {}".format(column)}
          }
    iplot(fig)


def stylistic_pie_chart(df, column):
    vals = list(df.groupby(column)['fully_funded'].value_counts())
    keys = list(df.groupby(column)['fully_funded'].value_counts().keys())
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

    trace = go.Pie(labels=keys, values=vals,
                   hoverinfo='label+percent', textinfo='value',
                   textfont=dict(size=20),
                   marker=dict(colors=colors,
                               line=dict(color='#000000', width=2)))

    iplot([trace], filename='styled_pie_chart')


def scatter_plot(df, x, y):
	'''
	Create scatter plot.

	Inputs:
		- df: dataframe
		- x: (str) attribute name
		- y: (str) attribute name
	'''
	plt.figure(figsize=(12,9))
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


def heatmap_correlation(df):
	corr = df.corr()
	plt.subplots(figsize=(20,15))
	mask = np.zeros_like(corr)
	mask[np.triu_indices_from(mask)] = True
	with sns.axes_style("white"):
		ax = sns.heatmap(corr, 
						 vmax=0.3,
						 mask=mask, 
						 square=True, 
						 linewidths=.5, 
						 annot=True, 
						 cmap="YlGnBu")


def find_most_funded(df, var_of_interest, groupby_var, top_n):
	'''
	Counts of top n most funded -- MAX n is 10
	'''
	return df.groupby(groupby_var)[var_of_interest].value_counts().nlargest(top_n)


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
	plt.tight_layout()
	plt.show()

def graph_mean_over_time(df1, df2, df3, X, Y):
    mean_df1 = df1.groupby(X).mean()
    mean_df2 = df2.groupby(X).mean()
    mean_df3 = df3.groupby(X).mean()

    x1 = list(mean_df1.index.values)
    y1 = list(mean_df1[Y].values)
    
    x2 = list(mean_df2.index.values)
    y2 = list(mean_df2[Y].values)
    
    x3 = list(mean_df3.index.values)
    y3 = list(mean_df3[Y].values)
    
    plt.plot(x1, y1)
    plt.title('Mean '+Y+' by '+X)
    plt.ylabel(Y)
    
    plt.plot(x2, y2)
    plt.title('Mean '+Y+' by '+X)
    plt.ylabel(Y)

    plt.plot(x3, y3)
    plt.title('Mean '+Y+' by '+X)
    plt.ylabel(Y)
    
    plt.show()


