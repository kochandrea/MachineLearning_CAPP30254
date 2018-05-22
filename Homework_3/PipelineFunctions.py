from __future__ import division


import LoadData as ld


import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
# %matplotlib inline
from sklearn.metrics import precision_recall_curve



'''
FUNCTIONS
'''
def temporal_splitter(df, features, target_var, num):
    '''
    features = list of features
    target = target variable
    num = refers to the dictionary keys of temporal_splits_dict
    ##With more time, would implement using datetime module
    '''
    
    temporal_splits_dict = {1:(('2011-01-01', '2011-12-31'),('2012-01-01', '2012-06-31')),
                        2:(('2011-01-01', '2012-06-31'),('2012-07-01', '2012-12-31')),
                        3:(('2011-01-01', '2012-12-31'),('2013-01-01', '2013-06-31')),
                        4:(('2011-01-01', '2013-06-31'),('2013-07-01', '2013-12-31'))}
    
    train_date_range = temporal_splits_dict[num][0]
    x_train = df[features + ['date_posted']]
    x_train = ld.limit_date_range(x_train, train_date_range, 'date_posted')
    print('x_train min:', min(x_train.date_posted),', x_train max: ', max(x_train.date_posted))
    x_train = x_train.drop('date_posted', axis=1)
    
    y_train = df[target_var + ['date_posted']]
    y_train = ld.limit_date_range(y_train, train_date_range, 'date_posted')
    print('y_train min:', min(y_train.date_posted),', y_train max: ', max(y_train.date_posted))
    y_train = y_train.drop('date_posted', axis=1)

    test_date_range = temporal_splits_dict[num][1]
    x_test = df[features + ['date_posted']]
    x_test = ld.limit_date_range(x_test, test_date_range, 'date_posted')
    print('x_test min:', min(x_test.date_posted),', x_test max: ', max(x_test.date_posted))
    x_test = x_test.drop('date_posted', axis=1)
    
    y_test = df[target_var + ['date_posted']]
    y_test = ld.limit_date_range(y_test, test_date_range, 'date_posted')
    print('y_test min:', min(y_test.date_posted),', y_test max: ', max(y_test.date_posted))
    y_test = y_test.drop('date_posted', axis=1)
    
    y_test = y_test[target_var]
    return x_train, x_test, y_train, y_test



'''
Below adapted from: https://github.com/rayidghani/magicloops/blob/master/simpleloop.py
'''

def F_score_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = np.asarray(generate_binary_at_k(y_scores, k))
    precision = precision_score(y_true, preds_at_k, labels=None)

    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)

    F_score = 2*((precision*recall)/(precision+recall))

    return F_score


def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3),
            'BAGGING': BaggingClassifier()
           }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
        'BAGGING':{}
           }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           ,'BAGGING':{}}
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           ,'BAGGING':{}}
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

# a set of helper function to do machine learning evalaution


def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = np.asarray(generate_binary_at_k(y_scores, k))
    #pdb.set_trace() to debug
    precision = precision_score(y_true, preds_at_k, labels=None)
    return precision


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    plt.show()
   

def recall_at_k(y_true, y_scores, k):
    '''
    From Rayid's mlfunctions.py: https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


def get_subsets(l):
    '''    
    From Rayid's mlfunctions.py: https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    subsets = []
    for i in range(1, len(l) + 1):
        for combo in itertools.combinations(l, i):
            subsets.append(list(combo))
    return subsets

def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test):
    '''
    Runs the loop using models_to_run, clfs, gridm and the data
    '''
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc', 'baseline',
                                        'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20',
                                        'p_at_30', 'p_at_50',
                                        'r_at_1', 'r_at_2', 'r_at_5', 'r_at_10', 'r_at_20',
                                        'r_at_30', 'r_at_50',
                                        'f_at_1', 'f_at_2', 'f_at_5', 'f_at_10', 'f_at_20',
                                        'f_at_30', 'f_at_50'))
                                        
    for n in range(1, 2):
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train.values.ravel()).predict_proba(X_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    y_pred_probs_sorted = np.asarray(y_pred_probs_sorted)
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,100.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 1.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 2.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 30.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted, 50.0),
                                                       F_score_at_k(y_test_sorted,y_pred_probs_sorted, 1.0),
                                                       F_score_at_k(y_test_sorted,y_pred_probs_sorted, 2.0),
                                                       F_score_at_k(y_test_sorted,y_pred_probs_sorted, 5.0),
                                                       F_score_at_k(y_test_sorted,y_pred_probs_sorted, 10.0),
                                                       F_score_at_k(y_test_sorted,y_pred_probs_sorted, 20.0),
                                                       F_score_at_k(y_test_sorted,y_pred_probs_sorted, 30.0),
                                                       F_score_at_k(y_test_sorted,y_pred_probs_sorted, 50.0)]
                    plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df

 
def go_function(X_train, X_test, y_train, y_test):

    # define grid to use: test, small, large
    grid_size = 'test'
    clfs, grid = define_clfs_params(grid_size)

    # define models to run
    models_to_run=['RF','DT','KNN', 'AB', 'LR', 'NB', 'BAGGING']

    #change output display in pandas
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # call clf_loop and store results in results_df
    results_df = clf_loop(models_to_run, clfs,grid, X_train, X_test, y_train, y_test)
    return results_df