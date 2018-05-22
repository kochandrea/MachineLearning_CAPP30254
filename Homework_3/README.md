Pipeline.ipynb:
- Notebook containing 

LoadData.py:
- Reads Kaggle data in pandas dataframes
- Merges outcomes and projects dataframes into donors_choose dataframe, limited to data between 2011 and 2013
- Stores dataframes as global variables

Variables.py:
- Stores lists of dataframe attribues as global variables, sorted by type of variable (eg. continuous, categorical)

ExplorationFunctions.py:
- Contains functions used to clean and explore the data
- Contains really cool donut and stylized pie plots which can only be viewed when in jupyter notebook

PipelineFunctions.py:
- Contains functions for evaluation metrics, classifier loop, and function for creating temporal train/test splits of data
- With more time, would format the temporal_splitter to employ datetime module