#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  11 12:04:23 2021

@author: daniel
"""
import sys 
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
import numpy as np
import random

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest

from boruta import BorutaPy
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_validate
from skopt import BayesSearchCV, plots
from skopt.plots import plot_convergence, plot_objective

"""
The RF imputation procedures improve performance if the features are heavily correlated.
!!! Correlation is important for RF imputation (https://arxiv.org/pdf/1701.05305.pdf) !!!
"""

def hyper_opt(classifier, data_x, data_y, n_iter=15, k_fold=10, params=None):
    """
    Optimizes hyperparameters using a k-fold cross validation splitting strategy.

    Example:
        We can explore any parameter space by creating a dictionary with the
        desired hyperparameteres:
        
        >>> classifier = RandomForestClassifier()
        >>> params = {'criterion': ["gini", "entropy"],
                     'n_estimators': [500,750,1000]}
        >>> model, params = hyper_opt(classifier, data_x, data_y) 
        
        The first output is our optimal classifier, and will be used to make predictions:
        
        >>> prediction = model.predict(new_data)
        
        The second output of the optimize function is the dictionary containing
        the hyperparameter combination that yielded the highest mean accuracy.
        
    Args:
        classifier: The machine learning classifier to optimize.
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        n_iter (int, optional): The maximum number of iterations to perform during 
            the Bayesian search.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.
        params (dict, optional): If None, the optimization procedure will cover
            a range of pre-configured hyperparameters. A dictionary with a custom
            range of hyperparameters can be input instead.
        
    Returns:
        The first output is the Random Forest classifier with the optimal hyperparameters.
        Second output is a dictionary containing the optimal hyperparameters.
    """
    
    if params is None:
        params = {
           'criterion': ["gini", "entropy"],
           'n_estimators': [int(x) for x in np.linspace(50,1000, num=20)], 
           'max_features': [len(data_x[0]), "sqrt", "log2"],
           'max_depth': [int(x) for x in np.linspace(5,50,num=5)],
           'min_samples_split': [3,4,6,7,8,9,10],
           'min_samples_leaf': [1,3,5,7,9,10],
           'max_leaf_nodes': [int(x) for x in np.linspace(2,200)],
           'bootstrap': [True,False]   
        }
        
    gs = BayesSearchCV(n_iter=15, estimator=classifier, search_spaces=params, optimizer_kwargs={'base_estimator': 'RF'}, cv=k_fold)
    print('Performing Bayesian optimization...')
    gs.fit(data_x, data_y)
    best_est = gs.best_estimator_
    best_score = np.round(gs.best_score_, 3)
    print(f"Highest mean accuracy: {best_score}")
    #plot_convergence(gs.optimizer_results_[0])
    #plot_objective(gs.optimizer_results_[0])
    #plots.plot_evaluations(gs.optimizer_results_[0])
    
    return gs.best_estimator_, gs.best_params_

def boruta_opt(classifier, data_x, data_y, k_fold=10):
    """
    Applies the Boruta algorithm (Kursa & Rudnicki 2011) to identify features
    that perform worse than random chance.
    See: https://arxiv.org/pdf/1106.5112.pdf

    Args:
        classifier: The machine learning classifier to use. It must have a method
            that outpouts the feature_importances_ attribute, like the Random Forest.
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        k_fold (int, optional): The number of cross-validations to perform.
            The output confusion matrix will display the mean accuracy across
            all k_fold iterations. Defaults to 10.
            
    Returns:
        1D array containing the indices of the selected features. This can then
        be used to index the columns in the data_x array.
    """

    feat_selector = BorutaPy(classifier, n_estimators='auto', verbose=0, random_state=1)
    print('Running feature optimization...')
    feat_selector.fit(data_x, data_y)

    feat_selector.support = np.array([str(feat) for feat in feat_selector.support_])
    index = np.where(feat_selector.support == 'True')[0]
    print('Feature selection complete, {} selected out of {}! Calculating classification accuracy...'.format(len(index),len(feat_selector.support)))
    
    cv = cross_validate(classifier, data_x[:,index], data_y, cv=k_fold)   
    print('Mean accuracy after {}-fold cross-validation: {}'.format(k_fold,np.round(np.mean(cv['test_score']),3)))
    
    return index

def Strawman_imputation(data):
    """
    Perform Strawman imputation, a time-efficient algorithm
    in which missing data values are replaced with the median
    value of the entire, non-NaN sample. If the data is a hot-encoded
    boolean (as the RF does not allow True or False), then the 
    instance that is used the most will be computed as the median. 

    This is the baseline algorithm used by (Tang & Ishwaran 2017).
    See: https://arxiv.org/pdf/1701.05305.pdf

    Note:
        This function assumes each row corresponds to one sample, and 
        that missing values are masked as either NaN or inf. 

    Args:
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.

    Returns:
        The data array with the missing values filled in. 
    """

    data[data>1e6] = 1e6
    data[(data>0) * (data<1e-6)] = 1e-6

    if np.all(np.isfinite(data)):
        print('No missing values in data, returning original array.')
        return data 

    if data.shape == 1:
        imputed_data = data
        mask = np.where(np.isfinite(data))
        median = np.median(data[mask])
        imputed_data[np.isnan(imputed_data) == True] = median 

        return imputed_data

    Nx = data.shape[1]
    Ny = data.shape[0] #Python reverses x & y values, y is the first axis
    imputed_data = np.zeros((Ny,Nx))
    for i in range(Nx):
        mask = np.where(np.isfinite(data[:,i]))
        median = np.median(data[:,i][mask])

        for j in range(Ny):
            if np.isnan(data[j,i]) == True or np.isinf(data[j,i]) == True:
                imputed_data[j,i] = median
            else:
                imputed_data[j,i] = data[j,i]

    return imputed_data 

def KNN_imputation(data, imputer=None, k=3):
    """
    Performs k-Nearest Neighbor imputation and transformation.
    By default the imputer will be created and returned, unless
    the imputer argument is set, in which case only the transformed
    data is output. 

    As this bundles neighbors according to their eucledian distance,
    it is sensitive to outliers. Can also yield weak predictions if the
    training features are heaviliy correlated.
    
    Args:
        imputer (optional): A KNNImputer class instance, configured using sklearn.impute.KNNImputer.
            Defaults to None, in which case the transformation is created using
            the data itself. 
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.
        k (int, optional): If imputer is None, this is the number
            of nearest neighbors to consider when computing the imputation.
            Defaults to 3. If imputer argument is set, this variable is ignored.

    Note:
        Tang & Ishwaran 2017 reported that if there is low to medium
        correlation in the dataset, RF imputation algorithms perform 
        better than KNN imputation

    Example:
        If we have our training data in an array called training_set, we 
        can create the imputer so that we can call it to transform new data
        when making on-the-field predictions.

        >>> imputed_data, knn_imputer = KNN_imputation(data=training_set, imputer=None)
        
        Now we can use the imputed data to create our machine learning model.
        Afterwards, when new data is input for prediction, we will insert our 
        imputer into the pipelinen by calling this function again, but this time
        with the imputer argument set:

        >>> new_data = knn_imputation(new_data, imputer=knn_imputer)

    Returns:
        The first output is the data array with with the missing values filled in. 
        The second output is the KNN Imputer that should be used to transform
        new data, prior to predictions. 
    """

    data[data>1e6] = 1e6
    data[(data>0) * (data<1e-6)] = 1e-6

    if np.all(np.isfinite(data)) and imputer is None:
        raise ValueError('No missing values in training dataset, do not apply imputation algorithms!')

    if imputer is None:
        imputer = KNNImputer(n_neighbors=k)
        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 

def MissForest_imputation(data, imputer=None):
    """
    Imputation algorithm created by Stekhoven and Buhlmann (2012).
    See: https://academic.oup.com/bioinformatics/article/28/1/112/219101

    By default the imputer will be created and returned, unless
    the imputer argument is set, in which case only the transformed
    data is output. 
    
    Args: 
        data (ndarray): 1D array if single parameter is input. If
            data is 2-dimensional, the medians will be calculated
            using the non-missing values in each corresponding column.
        imputer (optional): A MissForest class instance, configured using 
            the missingpy API. Defaults to None, in which case the transformation 
            is created using the data itself.

    Example:
        If we have our training data in an array called training_set, we 
        can create the imputer so that we can call it to transform new data
        when making on-the-field predictions.

        >>> imputed_data, rf_imputer = MissForest_imputation(data=training_set, imputer=None)
        
        Now we can use the imputed data to create our machine learning model.
        Afterwards, when new data is input for prediction, we will insert our 
        imputer into the pipelinen by calling this function again, but this time
        with the imputer argument set:

        >>> new_data = MissForest_imputer(new_data, imputer=rf_imputer)

    Returns:
        The first output is the data array with with the missing values filled in. 
        The second output is the Miss Forest Imputer that should be used to transform
        new data, prior to predictions. 
    """
    data[data>1e6] = 1e6
    data[(data>0) * (data<1e-6)] = 1e-6

    if np.all(np.isfinite(data)) and imputer is None:
        raise ValueError('No missing values in training dataset, do not apply imputation algorithms!')

    if imputer is None:
        imputer = MissForest(verbose=0)
        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 

