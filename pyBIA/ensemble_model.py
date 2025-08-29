#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 8 10:04:23 2021

@author: daniel
"""
import os
import copy 
import joblib 
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from cycler import cycler
from warnings import warn
from pathlib import Path
from collections import Counter  
from contextlib import suppress

from sklearn import decomposition
from xgboost import XGBClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import confusion_matrix, auc, RocCurveDisplay
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
#from scikitplot.metrics import plot_roc
from sklearn.manifold import TSNE

from optuna.importance import get_param_importances, FanovaImportanceEvaluator
from pyBIA.optimization import hyper_opt, borutashap_opt, impute_missing_values

#from lightgbm import LGBMClassifier


with suppress(ModuleNotFoundError):
    import scienceplots
    plt.style.use("science")
    plt.rcParams.update({"font.size": 21})


class Classifier:
    """
    Creates a machine-learning classifier with optional imputation, BorutaSHAP
    feature selection, and Optuna hyperparameter optimization. Utilities are
    provided to save/load artifacts and to plot diagnostics (t-SNE, confusion
    matrix, ROC, optimization history, and importances).

    Parameters
    ----------
    data_x : ndarray
        Feature matrix of shape (n_samples, n_features).
    data_y : array-like
        1D array of labels aligned to `data_x`.
    clf : str
        Estimator to build. One of {'rf','nn','xgb','histgb','adaboost','svc',
        'logreg','bdt','gaussian_nb','knn','extratrees','tree','ocsvm'}.
        Defaults to 'rf'.
    optimize : bool
        Run BorutaSHAP (when `boruta_trials` > 0) and Optuna search before fitting.
        Defaults to False.
    opt_cv : int
        Number of cross-validation folds used during optimization. Defaults to 10.
    scoring_metric : str
        Metric optimized by Optuna. One of {'accuracy','f1','precision','recall','roc_auc'}.
        Defaults to 'f1'.
    limit_search : bool
        Constrain very wide hyperparameter ranges for practicality. Defaults to True.
    impute : bool
        Impute missing values prior to fitting. Defaults to True.
    imp_method : str
        Imputation strategy. One of {'knn','mean','median','mode','constant'}.
        Defaults to 'knn'.
    n_iter : int
        Number of Optuna trials; use 0 to skip search. Defaults to 25.
    boruta_trials : int
        Number of BorutaSHAP trials; use 0 to skip feature selection. Defaults to 50.
    boruta_model : str
        Base estimator for BorutaSHAP, independent of `clf`. One of {'rf','xgb'}.
        Defaults to 'rf'.
    balance : bool
        Apply class weighting for imbalanced binary tasks where supported.
        Defaults to True.
    csv_file : DataFrame, optional
        Alternative to (`data_x`, `data_y`). Must include a 'label' column.
        Defaults to None.
    SEED_NO : int
        Random seed used across components. Defaults to 1909.

    Attributes
    ----------
    data_x : ndarray or None
        Possibly imputed/processed feature matrix.
    data_y : ndarray or None
        Numeric labels used for fitting (may be encoded).
    data_y_ : ndarray or None
        Copy of original labels (pre-encoding) for plots.
    clf : str
        Name of the chosen estimator.
    model : estimator or None
        Trained estimator instance.
    imputer : object or None
        Fitted imputer used for transformations.
    feats_to_use : ndarray or None
        Indices of selected features (BorutaSHAP).
    feature_history : object or None
        BorutaSHAP selection history.
    optimization_results : optuna.study.Study or None
        Study from hyperparameter search.
    best_params : dict or None
        Best hyperparameters from Optuna.
    path : str or None
        Directory used when saving artifacts.
    SEED_NO : int
        Seed propagated to internal routines.
    """

    def __init__(
        self, 
        data_x=None, 
        data_y=None, 
        clf='rf', 
        optimize=False, 
        opt_cv=10, 
        scoring_metric='f1',
        limit_search=True, 
        impute=True, 
        imp_method='knn', 
        n_iter=25, 
        boruta_trials=50, 
        boruta_model='rf', 
        balance=True, 
        csv_file=None, 
        SEED_NO=1909
        ):

        self.data_x = data_x
        self.data_y = data_y
        self.clf = clf
        self.optimize = optimize 
        self.opt_cv = opt_cv 
        self.scoring_metric = scoring_metric
        self.limit_search = limit_search
        self.impute = impute
        self.imp_method = imp_method
        self.n_iter = n_iter
        self.boruta_trials = boruta_trials
        self.boruta_model = boruta_model 
        self.balance = balance 
        self.csv_file = csv_file
        self.SEED_NO = SEED_NO

        self.model = None
        self.imputer = None
        self.feats_to_use = None

        self.feature_history = None  
        self.optimization_results = None 
        self.best_params = None 

        if self.csv_file is not None:
            self.data_x = np.array(csv_file[csv_file.columns[:-1]])
            self.data_y = csv_file.label
            print('Successfully loaded the data_x and data_y arrays from the input csv_file!')
        else:
            if self.data_x is None or self.data_y is None:
                print('NOTE: data_x and data_y parameters are required to output visualizations.')
        
        if self.data_y is not None:
            self.data_y_ = copy.deepcopy(self.data_y) #For plotting purposes, save the original label array as it will be overwritten with the numerical labels when plotting
            if self.clf == 'xgb':
                if all(isinstance(val, (int, str)) for val in self.data_y):
                    print('XGBoost classifier requires numerical class labels! Converting class labels as follows:')
                    print('________________________________')
                    y = np.zeros(len(self.data_y))
                    for i in range(len(np.unique(self.data_y))):
                        print(str(np.unique(self.data_y)[i]).ljust(10)+'  ------------->     '+str(i))
                        index = np.where(self.data_y == np.unique(self.data_y)[i])[0]
                        y[index] = i
                    self.data_y = y 
                    print('________________________________')
        else:
            self.data_y_ = None 

    def create(self, overwrite_training=True):
        """
        Builds the pipeline (optional feature selection and optimization), fits the
        estimator, and stores artifacts.

        Parameters
        ----------
        overwrite_training : bool
            When True, replace `self.data_x` with the processed matrix used for
            fitting. Defaults to True.

        Returns
        -------
        None
        """

        if self.optimize is False:
            if len(np.unique(self.data_y)) == 2:
                counter = Counter(self.data_y)
                if counter[np.unique(self.data_y)[0]] != counter[np.unique(self.data_y)[1]]:
                    if self.balance: #If balance is True but optimize is False
                        print('Unbalanced dataset detected, to apply weights set optimize=True.')

        if self.clf == 'rf':
            model = RandomForestClassifier(random_state=self.SEED_NO)
        elif self.clf == 'nn':
            model = MLPClassifier(max_iter=1000, early_stopping=True, random_state=self.SEED_NO)
        elif self.clf == 'histgb':
            model = HistGradientBoostingClassifier(random_state=self.SEED_NO)
        elif self.clf == 'adaboost':
            model = AdaBoostClassifier(random_state=self.SEED_NO)
        elif self.clf == 'svc':
            model = SVC(probability=True, random_state=self.SEED_NO)
        elif self.clf == 'logreg':
            model = LogisticRegression(random_state=self.SEED_NO)
        elif self.clf == 'xgb':
            model = XGBClassifier(random_state=self.SEED_NO)
        elif self.clf == 'bdt':
            model = GradientBoostingClassifier(random_state=self.SEED_NO)
        elif self.clf == 'gaussian_nb':
            model = GaussianNB() # No seed required as this algo is deterministic!
        elif self.clf == 'knn':
            model = KNeighborsClassifier() # No seed required as this algo is deterministic!
        elif self.clf == 'extratrees':
            model = ExtraTreesClassifier(random_state=self.SEED_NO)
        elif self.clf == 'tree':
            model = DecisionTreeClassifier(random_state=self.SEED_NO)
        elif self.clf == 'ocsvm':
            if self.data_y is not None:
                if len(np.unique(self.data_y)) != 1:
                    raise ValueError('The clf parameter has been set to "ocsvm" but OneClassSVM requires that only the positive class be input!')
            model = OneClassSVM() # No seed required as this algo is deterministic!
        else:
            raise ValueError('Invalid clf argument!')
        #
        if all(isinstance(val, (int, str)) for val in self.data_y):
            print('XGBoost classifier requires numerical class labels! Converting class labels as follows:')
            print('________________________________')
            y = np.zeros(len(self.data_y))
            for i in range(len(np.unique(self.data_y))):
                print(str(np.unique(self.data_y)[i]).ljust(10)+'  ------------->     '+str(i))
                index = np.where(self.data_y == np.unique(self.data_y)[i])[0]
                y[index] = i
            self.data_y = y 
            print('________________________________')

        self.data_x[np.isinf(self.data_x)] = np.nan

        if self.impute is False and self.optimize is False:
            #data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
            if np.any(np.isfinite(self.data_x)==False):
                raise ValueError('data_x array contains nan values but impute is set to False! Set impute=True and run again.')
            print("Returning base {} model...".format(self.clf))
            model.fit(self.data_x, self.data_y)
            self.model = model
            #self.data_x = data if overwrite_training else self.data_x

            return

        if self.impute:
            data, self.imputer = impute_missing_values(self.data_x, strategy=self.imp_method)
            #data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
            if self.optimize is False:
                #data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
                print("Returning base {} model...".format(self.clf))
                model.fit(data, self.data_y)
                self.model = model 
                self.data_x = data if overwrite_training else self.data_x

                return
        else:
            data = copy.deepcopy(self.data_x)
            #data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7

        if self.feats_to_use is None:
            self.feats_to_use, self.feature_history = borutashap_opt(data, self.data_y, boruta_trials=self.boruta_trials, model=self.boruta_model, SEED_NO=self.SEED_NO)
            if len(self.feats_to_use) == 0:
                print('No features selected, increase the number of n_trials when running pyBIA.optimization.borutashap_opt(). Using all features...')
                self.feats_to_use = np.arange(data.shape[1])
        else:
            print('The feats_to_use attribute already exists, skipping feature selection...')

        #Re-construct the imputer with the selected features as new predictions will only compute these metrics, so need to fit again!
        if self.impute:
            data_x, self.imputer = impute_missing_values(self.data_x[:,self.feats_to_use], strategy=self.imp_method)
        else:
            data_x, self.imputer = self.data_x[:,self.feats_to_use], None

        if self.n_iter > 0:
            self.model, self.best_params, self.optimization_results = hyper_opt(data_x, self.data_y, clf=self.clf, n_iter=self.n_iter, 
                balance=self.balance, return_study=True, limit_search=self.limit_search, opt_cv=self.opt_cv, scoring_metric=self.scoring_metric, 
                SEED_NO=self.SEED_NO)
        else:
            print("Fitting and returning final model...")
            self.model = hyper_opt(data_x, self.data_y, clf=self.clf, n_iter=self.n_iter, balance=self.balance, return_study=True, limit_search=self.limit_search, 
                scoring_metric=self.scoring_metric, opt_cv=self.opt_cv, SEED_NO=self.SEED_NO)

        self.model.fit(data_x, self.data_y)
        self.data_x = data if overwrite_training else self.data_x

        return
        
    def save(self, dirname=None, path=None, overwrite=False):
        """
        Saves the trained model and auxiliary artifacts.

        Notes
        -----
        Creates a `pyBIA_ensemble_model/` folder containing, when available:
        `Model`, `Imputer`, `Feats_Index`, `HyperOpt_Results`, `Best_Params`,
        and `FeatureOpt_Results`.

        Parameters
        ----------
        dirname : str, optional
            Subdirectory name created under `path`. Defaults to None.
        path : str, optional
            Base directory for saving. The user home is used when not provided.
            Defaults to None.
        overwrite : bool
            Remove any existing `pyBIA_ensemble_model` at the target before saving.
            Defaults to False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If nothing has been created (run `.create()` first) or if the target
            exists and `overwrite` is False.
        """

        if self.model is None and self.imputer is None and self.feats_to_use is None:
            raise ValueError('The models have not been created! Run the create() method first.')

        path = str(Path.home()) if path is None else path 
        path = path + '/' if path[-1] != '/' else path 
        
        if dirname is not None:
            dirname = dirname + '/' if dirname[-1] != '/' else dirname
            path = path + dirname
            try:
                os.makedirs(path)
            except FileExistsError:
                raise ValueError('The dirname folder already exists!')

        try:
            os.mkdir(path + 'pyBIA_ensemble_model')
        except FileExistsError:
            if overwrite:
                try:
                    os.rmdir(path+'pyBIA_ensemble_model')
                except OSError:
                    for file in os.listdir(path+'pyBIA_ensemble_model'):
                        os.remove(path+'pyBIA_ensemble_model/'+file)
                    os.rmdir(path+'pyBIA_ensemble_model')
                os.mkdir(path+'pyBIA_ensemble_model')
            else:
                raise ValueError('Tried to create "pyBIA_ensemble_model" directory in specified path but folder already exists! If you wish to overwrite set overwrite=True.')
        
        path += 'pyBIA_ensemble_model/'
        if self.model is not None:
            joblib.dump(self.model, path+'Model')
        if self.imputer is not None:
            joblib.dump(self.imputer, path+'Imputer')
        if self.feats_to_use is not None:
            joblib.dump(self.feats_to_use, path+'Feats_Index')
        if self.optimization_results is not None:
            joblib.dump(self.optimization_results, path+'HyperOpt_Results')
        if self.best_params is not None:
            joblib.dump(self.best_params, path+'Best_Params')
        if self.feature_history is not None:
            joblib.dump(self.feature_history, path+'FeatureOpt_Results')

        print('Files saved in: {}'.format(path))

        self.path = path

        return 

    def load(self, path=None):
        """
        Loads model and auxiliary artifacts from a `pyBIA_ensemble_model/` folder.

        Parameters
        ----------
        path : str, optional
            Base directory containing the folder. The user home is used when not
            provided. Defaults to None.

        Returns
        -------
        None
        """

        path = str(Path.home()) if path is None else path 
        path = path+'/' if path[-1] != '/' else path 
        path += 'pyBIA_ensemble_model/'

        try:
            self.model = joblib.load(path+'Model')
            model = 'model'
        except FileNotFoundError:
            model = ''
            pass

        try:
            self.imputer = joblib.load(path+'Imputer')
            imputer = ', imputer'
        except FileNotFoundError:
            imputer = ''
            pass 

        try:
            self.feats_to_use = joblib.load(path+'Feats_Index')
            feats_to_use = ', feats_to_use'
        except FileNotFoundError:
            feats_to_use = ''
            pass

        try:
            self.best_params = joblib.load(path+'Best_Params')
            best_params = ', best_params'
        except FileNotFoundError:
            best_params = ''
            pass

        try:
            self.feature_history = joblib.load(path+'FeatureOpt_Results')
            feature_opt_results = ', feature_selection_results'
        except FileNotFoundError:
            feature_opt_results = ''
            pass

        try:
            self.optimization_results = joblib.load(path+'HyperOpt_Results')
            optimization_results = ', optimization_results'
        except FileNotFoundError:
            optimization_results = '' 
            pass

        print('Successfully loaded the following class attributes: {}{}{}{}{}{}'.format(model, imputer, feats_to_use, best_params, feature_opt_results, optimization_results))
        
        self.path = path

        return

    def predict(self, data):
        """
        Predicts class labels and top-class probabilities for new samples.

        Parameters
        ----------
        data : ndarray
            Feature matrix of shape (n_samples, n_features). If feature selection
            was used, only the selected columns are required.

        Returns
        -------
        ndarray
            Array of shape (n_samples, 2) with rows
            [predicted_label, probability_of_predicted_label].
        """

        #data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
        classes = self.model.classes_
        output = []

        if self.imputer is None and self.feats_to_use is None:
            proba = self.model.predict_proba(data)
            for i in range(len(proba)):
                index = np.argmax(proba[i])
                output.append([classes[index], proba[i][index]])
            return np.array(output)

        if self.feats_to_use is not None:
            data = data[self.feats_to_use].reshape(1,-1) if len(data.shape) == 1 else data[:,self.feats_to_use]
            data = self.imputer.transform(data) if self.imputer is not None else data
            proba = self.model.predict_proba(data)

            for i in range(len(proba)):
                index = np.argmax(proba[i])
                output.append([classes[index], proba[i][index]])

            return np.array(output)

        data = self.imputer.transform(data) if self.imputer is not None else data
        proba = self.model.predict_proba(data)

        for i in range(len(proba)):
            index = np.argmax(proba[i])
            output.append([classes[index], proba[i][index]])
            
        return np.array(output)

    def plot_tsne(
        self, 
        data_y=None, 
        special_class=None, 
        norm=True, 
        pca=False, 
        return_data=False,
        xlim=None, 
        ylim=None, 
        legend_loc='upper center', 
        title='Feature Parameter Space', 
        savefig=False
        ):
        """
        Plots a 2D t-SNE embedding of the feature space.

        Parameters
        ----------
        data_y : array-like, optional
            Labels for coloring. The classifier’s labels are used when not provided.
            Defaults to None.
        special_class : hashable, optional
            Class label to highlight. Defaults to None.
        norm : bool
            Standardize features before t-SNE. Defaults to True.
        pca : bool
            Apply PCA (all components) before t-SNE. Defaults to False.
        return_data : bool
            Return the (x, y) coordinates instead of only plotting. Defaults to False.
        xlim : tuple, optional
            X-axis limits. Defaults to None.
        ylim : tuple, optional
            Y-axis limits. Defaults to None.
        legend_loc : str
            Legend location. Defaults to 'upper center'.
        title : str
            Figure title. Defaults to 'Feature Parameter Space'.
        savefig : bool
            Save a PNG instead of showing. Defaults to False.

        Returns
        -------
        AxesImage or tuple
            When `return_data` is False, returns the plotted artist. When True,
            returns `(x, y)` coordinates.
        """

        if self.feats_to_use is not None:
            data = self.data_x[self.feats_to_use].reshape(1,-1) if len(self.data_x.shape) == 1 else self.data_x[:,self.feats_to_use] 
        else:
            data = copy.deepcopy(self.data_x)

        if np.any(np.isnan(data)):
            data = impute_missing_values(data, self.imputer) if self.imputer is not None else impute_missing_values(data, strategy=self.imp_method)[0]
            
        #data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7
        
        method = 'barnes_hut' if len(data) > 5e3 else 'exact' #bh Scales with O(N), exact scales with O(N^2)

        if norm:
            #from sklearn.preprocessing import PowerTransformer
            #scaler = PowerTransformer(method='yeo-johnson')

           # scaler = MinMaxScaler()
            scaler = StandardScaler()
            #scaler = RobustScaler()
            data = scaler.fit_transform(data)

        if pca:
            pca_transformation = decomposition.PCA(n_components=data.shape[1], whiten=True, svd_solver='auto')
            pca_transformation.fit(data) 
            data = pca_transformation.transform(data)

       # feats = TSNE(n_components=2, method=method, learning_rate='auto', perplexity=15, init='pca', random_state=self.SEED_NO).fit_transform(data)
        #feats = TSNE(n_components=2, method=method, perplexity=300, init='pca', n_jobs=-1, random_state=self.SEED_NO).fit_transform(data)
        feats = TSNE(n_components=2, method=method, perplexity=150, init='pca', n_jobs=-1, random_state=self.SEED_NO).fit_transform(data)
        x, y = feats[:,0], feats[:,1]

        #from umap import UMAP
        #print('filt')
        #feats = UMAP(random_state=self.SEED_NO).fit_transform(data)
        #x, y = feats[:, 0], feats[:, 1]

     
        markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
        #color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#e41a1c', '#377eb8'] #Update the last two!

        _set_style_() if savefig else plt.style.use('default')
        
        if data_y is None:
            if self.data_y_ is None:
                if self.csv_file is None:
                    if data_y is None:
                        data_y = self.data_y
                        feats = np.unique(self.data_y)
                    else:
                        if isinstance(data_y, np.ndarray) is False: 
                            if type(data_y) == list:
                                data_y = np.array(data_y)
                            else:
                                raise ValueError('data_y argument must either be a list or an array!')
                        feats = np.unique(data_y)
                else:
                    data_y = np.array(self.csv_file.label)
                    feats = np.unique(data_y)
            else:
                data_y = self.data_y_ 
                feats = np.unique(self.data_y_)
        else:
            if isinstance(data_y, list):
                data_y = np.array(data_y)
            feats = np.unique(data_y) 

        for count, feat in enumerate(feats):
            if count+1 > len(markers):
                count = -1
            mask = np.where(data_y == feat)[0]
            if feat == special_class:
                pass
            else:
                plt.scatter(x[mask], y[mask], marker=markers[count], c=color[count], label=str(feat), alpha=0.44)

        if special_class is not None:
            mask = np.where(data_y == special_class)[0]
            if len(mask) == 0:
                raise ValueError('The data_y array does not contain the value input in the special_class parameter.')
            plt.scatter(x[mask], y[mask], marker='*', c='red', label=special_class, s=200, alpha=1.0)
        
        plt.xlim((xlim)) if xlim is not None else None 
        plt.ylim((ylim)) if ylim is not None else None 
        plt.legend(loc=legend_loc, ncol=len(np.unique(data_y)), frameon=False, handlelength=2)
        plt.title(title); plt.ylabel('t-SNE Dimension 1'); plt.xlabel('t-SNE Dimension 2')
        plt.xticks(); plt.yticks()

        if savefig:
            plt.savefig('tSNE_Projection.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        if return_data:
            return x, y
        else:
            return

    def plot_conf_matrix(
        self, 
        data_y=None, 
        norm=False, 
        pca=False, 
        k_fold=10, 
        normalize=True, 
        title='Confusion Matrix', 
        savefig=False
    ):
        """
        Plots a confusion matrix under k-fold cross-validation.

        Parameters
        ----------
        data_y : array-like, optional
            Human-readable labels aligned to the model’s internal labels. The
            classifier’s labels are used when not provided. Defaults to None.
        norm : bool
            Min-max normalize features before evaluation. Defaults to False.
        pca : bool
            Evaluate on PCA-projected features. Defaults to False.
        k_fold : int
            Number of cross-validation folds. Defaults to 10.
        normalize : bool
            Show rates (True) or counts (False). Defaults to True.
        title : str
            Figure title. Defaults to 'Confusion Matrix'.
        savefig : bool
            Save a PNG instead of showing. Defaults to False.

        Returns
        -------
        AxesImage
        """

        if self.data_x is None or self.data_y is None:
            raise ValueError('The data_x and data_y have not been input!')
        if self.model is None:
            raise ValueError('No model has been created! Run .create() first.')

        # To derive class names in the SAME order as numeric codes used during training
        def _classes_from_aligned_text(code_order, y_num, y_txt):
            y_num = np.asarray(y_num, dtype=int)
            y_txt = np.asarray(y_txt)
            names = []
            for c in code_order:
                mask = (y_num == int(c))
                if mask.any():
                    vals, cnts = np.unique(y_txt[mask], return_counts=True)
                    names.append(str(vals[np.argmax(cnts)]))
                else:
                    names.append(str(int(c)))  # fallback
            return names

        # Now choose the per-sample TEXT labels aligned to self.data_y 
        data_y_text = None
        if data_y is not None and len(data_y) == len(self.data_y):
            data_y_text = data_y
        elif getattr(self, "data_y_", None) is not None and len(self.data_y_) == len(self.data_y):
            data_y_text = self.data_y_
        elif getattr(self, "csv_file", None) is not None:
            try:
                lbls = np.array(self.csv_file.label)
                if len(lbls) == len(self.data_y):
                    data_y_text = lbls
            except Exception:
                pass

        if self.feats_to_use is not None:
            if len(self.data_x.shape) == 1:
                data = self.data_x[self.feats_to_use].reshape(1, -1)
            else:
                data = self.data_x[:, self.feats_to_use]
        else:
            data = copy.deepcopy(self.data_x)

        if np.any(np.isnan(data)):
            data = (impute_missing_values(data, self.imputer) if self.imputer is not None
                    else impute_missing_values(data, strategy=self.imp_method)[0])

        if norm:
            data = MinMaxScaler().fit_transform(data)

        if pca:
            pca_transformation = decomposition.PCA(n_components=data.shape[1], whiten=True, svd_solver='auto')
            pca_transformation.fit(data)
            data = np.asarray(pca_transformation.transform(data)).astype('float64')

        predicted_target, actual_target = evaluate_model(
            self.model, data, self.data_y, normalize=normalize, k_fold=k_fold, random_state=self.SEED_NO
        )
        actual_target = np.asarray(actual_target, dtype=int)

        code_order = np.sort(np.unique(actual_target))

        if data_y_text is not None:
            classes = _classes_from_aligned_text(code_order, self.data_y, data_y_text)
        else:
            classes = [str(int(c)) for c in code_order]

        return generate_matrix(
            predicted_target,
            actual_target,
            normalize=normalize,
            classes=classes,
            title=title,
            savefig=savefig
        )

    def plot_roc_curve(
        self, 
        k_fold=10, 
        pca=False, 
        title="Receiver Operating Characteristic Curve", 
        savefig=False
        ):
        """
        Plots the mean ROC curve with ±1σ band under k-fold cross-validation for
        binary classification.

        Parameters
        ----------
        k_fold : int
            Number of cross-validation folds. Defaults to 10.
        pca : bool
            Evaluate on PCA-projected features. Defaults to False.
        title : str
            Figure title. Defaults to "Receiver Operating Characteristic Curve".
        savefig : bool
            Save a PNG instead of showing. Defaults to False.

        Returns
        -------
        AxesImage
        """

        if self.model is None:
            raise ValueError('No model has been created! Run model.create() first.')

        if self.feats_to_use is not None:
            if len(self.data_x.shape) == 1:
                data = self.data_x[self.feats_to_use].reshape(1,-1)
            else:
                data = self.data_x[:,self.feats_to_use]
        else:
            data = copy.deepcopy(self.data_x)

        if np.any(np.isnan(data)):
            data = impute_missing_values(data, self.imputer) if self.imputer is not None else impute_missing_values(data, strategy=self.imp_method)[0]
          
        #data[data>1e7], data[(data<1e-7)&(data>0)], data[data<-1e7] = 1e7, 1e-7, -1e7

        if pca:
            pca_transformation = decomposition.PCA(n_components=data.shape[1], whiten=True, svd_solver='auto')
            pca_transformation.fit(data) 
            pca_data = pca_transformation.transform(data)
            data = np.asarray(pca_data).astype('float64')
        
        model0 = self.model
        if len(np.unique(self.data_y)) != 2:
            print("ROC Curves for more than two classes not currently supported!")
            #X_train, X_test, y_train, y_test = train_test_split(data, self.data_y, test_size=0.2, random_state=self.SEED_NO)
            #model0.fit(X_train, y_train)
            #y_probas = model0.predict_proba(X_test)
            #plot_roc(y_test, y_probas, text_fontsize='large', title='ROC Curve', cmap='cividis', plot_macro=False, plot_micro=False)
            #plt.show()
            return

        cv = StratifiedKFold(n_splits=k_fold)
        
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()

        for i, (data_x, test) in enumerate(cv.split(data, self.data_y)):
            model0.fit(data[data_x], self.data_y[data_x])
            viz = RocCurveDisplay.from_estimator(model0, data[test], self.data_y[test], alpha=0, lw=1, ax=ax, name="ROC fold {}".format(i+1))
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr); aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc, std_auc = auc(mean_fpr, mean_tpr), np.std(aucs)
        lns1, = ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean (AUC = %0.2f)" % (mean_auc), lw=2, alpha=0.8) #label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),

        std_tpr = np.std(tprs, axis=0)
        tprs_upper, tprs_lower = np.minimum(mean_tpr + std_tpr, 1), np.maximum(mean_tpr - std_tpr, 0)
        lns_sigma = ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1$\sigma$")

        ax.set(xlim=[0, 1.0], ylim=[0.0, 1.0], title="Receiver Operating Characteristic Curve")
        lns2, = ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random (AUC=0.5)", alpha=0.8)

        ax.legend([lns2, (lns1, lns_sigma)], ['Random (AUC = 0.5)', r"Mean (AUC = %0.2f)" % (mean_auc)], loc='lower center', ncol=2, frameon=False, handlelength=2)
        plt.title(label=title); plt.ylabel('True Positive Rate'); plt.xlabel('False Positive Rate')
        ax.set_facecolor("white")

        if savefig:
            _set_style_()
            plt.savefig('Ensemble_ROC_Curve.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def plot_hyper_opt(
        self, 
        baseline=None, 
        xlim=None, 
        ylim=None, 
        xlog=True, 
        ylog=False, 
        ylabel=None, 
        title=None, 
        loc='upper left', 
        ncol=1, 
        savefig=False
        ):
        """
        Visualizes Optuna optimization history: trial values and running best.

        Parameters
        ----------
        baseline : float, optional
            Horizontal baseline to compare against. Defaults to None.
        xlim : tuple, optional
            X-axis limits. Defaults to None.
        ylim : tuple, optional
            Y-axis limits. Defaults to None.
        xlog : bool
            Log-scale the x-axis. Defaults to True.
        ylog : bool
            Log-scale the y-axis. Defaults to False.
        ylabel : str, optional
            Custom y-axis label. Defaults to None.
        title : str, optional
            Custom title; inferred from `clf` when not set. Defaults to None.
        loc : str
            Legend location. Defaults to 'upper left'.
        ncol : int
            Number of legend columns. Defaults to 1.
        savefig : bool
            Save a PNG instead of showing. Defaults to False.

        Returns
        -------
        AxesImage
        """

        if savefig is False:
            plt.style.use('default')

        trials = self.optimization_results.get_trials()
        trial_values, best_value = [], []
        for trial in range(len(trials)):
            value = trials[trial].values[0]
            trial_values.append(value)
            if trial == 0:
                best_value.append(value)
            else:
                if any(y > value for y in best_value): #If there are any numbers in best values that are higher than current one
                    best_value.append(np.array(best_value)[trial-1])
                else:
                    best_value.append(value)

        best_value, trial_values = np.array(best_value), np.array(trial_values)
        best_value[1] = trial_values[1] #Make the first trial the best model, since technically it is.
        for i in range(2, len(trial_values)):
            if trial_values[i] < best_value[1]:
                best_value[i] = best_value[1]
            else:
                break

        plt.figure(figsize=(8,8))
        if baseline is not None:
            plt.axhline(y=baseline, color='k', linestyle='--', label='Baseline Model')

        plt.plot(range(1, len(trials)+1), best_value, color='r', alpha=0.83, linestyle='-', label='Optimized Model')
        plt.scatter(range(1, len(trials)+1), trial_values, c='b', marker='+', s=35, alpha=0.45, label='Trial')
        plt.xlabel('Trial Number', alpha=1, color='k')

        if ylabel is None:
            if self.opt_cv > 0:
                plt.ylabel(f'{scoring_metric} ({str(self.opt_cv)}-Fold Cross-Validation)', alpha=1, color='k')
            else:
                plt.ylabel(f'{scoring_metric}', alpha=1, color='k')
        else:
            plt.ylabel(ylabel, alpha=1, color='k')
        
        if title is None:
            if self.clf == 'xgb':
                plt.title('XGBoost Hyperparameter Optimization')
            elif self.clf == 'rf':
                plt.title('RF Hyperparameter Optimization')
            elif self.clf == 'ocsvm':
                plt.title('OneClass SVM Hyperparameter Optimization')
            elif self.clf == 'nn':
                plt.title('Neural Network Hyperparameter Optimization')
        else:
            plt.title(title)

        plt.legend(loc=loc, ncol=ncol, frameon=True, fancybox=True, handlelength=1)
        plt.rcParams['axes.facecolor']='white'
        plt.grid(False)

        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim((1, len(trials)+1))
        if ylim is not None:
            plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig('Ensemble_Hyperparameter_Optimization.png', bbox_inches='tight', dpi=300)
            plt.clf()#; plt.style.use('default')
        else:
            plt.show()

        return

    def plot_feature_opt(
        self, 
        feat_names=None, 
        top='all', 
        include_other=True, 
        include_shadow=True, 
        include_rejected=False, 
        flip_axes=True, 
        title='Feature Importance', 
        save_data=False, 
        savefig=False
        ):
        """
        Displays BorutaSHAP z-score distributions per feature across trials.

        Parameters
        ----------
        feat_names : array-like, optional
            Names for features in `data_x`. Defaults to None.
        top : int or 'all'
            Number of accepted features to show; 'all' shows every accepted feature.
            Defaults to 'all'.
        include_other : bool
            Aggregate remaining accepted features into an "Other Accepted" entry.
            Defaults to True.
        include_shadow : bool
            Include the Max Shadow baseline. Defaults to True.
        include_rejected : bool
            Append averaged rejected features. Defaults to False.
        flip_axes : bool
            Plot horizontally (True) or vertically (False). Defaults to True.
        title : str
            Figure title. Defaults to 'Feature Importance'.
        save_data : bool
            Keep the temporary CSV written by BorutaSHAP for this plot. Defaults to False.
        savefig : bool
            Save a PNG instead of showing. Defaults to False.

        Returns
        -------
        AxesImage
        """

        fname = str(Path.home()) + '/__borutaimportances__' #Temporary file

        try:
            self.feature_history.results_to_csv(filename=fname)
        except AttributeError:
            raise ValueError('No optimization history found for feature selection, run .create() with optimize=True!')

        csv_data = pd.read_csv(fname+'.csv')
        if save_data is False:
            os.remove(fname+'.csv')

        accepted_indices = np.where(csv_data.Decision == 'Accepted')[0]
        if top == 'all':
            top = len(accepted_indices)
        else:
            if top > len(accepted_indices):
                top = len(accepted_indices)
                print('The top parameter exceeds the number of accepted variables, setting to the maximum value of {}'.format(str(top)))

        x, y, y_err = [], [], []

        for i in accepted_indices[:top]:
            if feat_names is None:
                if self.csv_file is None:
                    x.append(int(i))
                else:
                    x.append(int(csv_data.iloc[i].Features))
            else:
                x.append(int(csv_data.iloc[i].Features))
            y.append(float(csv_data.iloc[i]['Average Feature Importance']))
            y_err.append(float(csv_data.iloc[i]['Standard Deviation Importance']))
            
        include_other = False if len(accepted_indices) == top else True

        if include_other:
            mean, std = [], []
            for j in accepted_indices[top:]:
                mean.append(float(csv_data.iloc[j]['Average Feature Importance']))
                std.append(float(csv_data.iloc[j]['Standard Deviation Importance']))
            x.append(0), y.append(np.mean(mean)), y_err.append(np.mean(std))

        if include_shadow:
            ix = np.where(csv_data.Features == 'Max_Shadow')[0]
            y.append(float(csv_data.iloc[ix]['Average Feature Importance']))
            y_err.append(float(csv_data.iloc[ix]['Standard Deviation Importance']))
            x.append(int(ix))

        if feat_names is not None:  
            feat_names = np.array(feat_names) if isinstance(feat_names, np.ndarray) is False else feat_names
            if include_shadow is False:
                x_names = feat_names[x] if include_other is False else np.r_[feat_names[x[:-1]], ['Other Accepted']] #By default x is the index of the feature
            else:
                x_names = np.r_[feat_names[x[:-1]], ['Max Shadow']] if include_other is False else np.r_[feat_names[x[:-2]], ['Other Accepted'], ['Max Shadow']]
        else:
            if self.csv_file is None:
                if include_other is False:
                    x_names = csv_data.iloc[x].Features if include_shadow is False else np.r_[csv_data.iloc[x[:-1]].Features, ['Max Shadow']]
                else:
                    x_names = np.r_[csv_data.iloc[x[:-1]].Features, ['Max Shadow']] if include_shadow is False else np.r_[csv_data.iloc[x[:-2]].Features, ['Other Accepted'], ['Max Shadow']]
            else:
                if include_other is False:
                    x_names = self.csv_file.columns[x[:-1]] if include_shadow is False else np.r_[self.csv_file.columns[x[:-1]], ['Max Shadow']]
                else:
                    x_names = np.r_[self.csv_file.columns[x[:-1]], ['Max Shadow']] if include_shadow is False else np.r_[self.csv_file.columns[x[:-2]], ['Other Accepted'], ['Max Shadow']]

        if include_rejected:
            x = []
            rejected_indices = np.where(csv_data.Decision == 'Rejected')[0]
            for i in rejected_indices:
                if feat_names is None:
                    if self.csv_file is None:
                        x.append(int(i))
                    else:
                        x.append(int(csv_data.iloc[i].Features))
                else:
                    x.append(int(csv_data.iloc[i].Features))
                y.append(float(csv_data.iloc[i]['Average Feature Importance']))
                y_err.append(float(csv_data.iloc[i]['Standard Deviation Importance']))

            if feat_names is None:
                x_names = np.r_[x_names, csv_data.iloc[x].Features] if self.csv_file is None else np.r_[x_names, self.csv_file.columns[x]]
            else:
                x_names = np.r_[x_names, feat_names[x]]
        
        y, y_err = np.array(y), np.array(y_err)

        fig, ax = plt.subplots(figsize=(8, 8))

        if flip_axes:
            lns, = ax.plot(y, np.arange(len(x_names)), 'k*--', lw=0.77)
            lns_sigma = ax.fill_betweenx(np.arange(len(x_names)), y-y_err, y+y_err, color="grey", alpha=0.2)
            ax.set_xlabel('Z Score', alpha=1, color='k'); ax.set_yticks(np.arange(len(x_names)), x_names)#, rotation=90)
            
            for t in ax.get_yticklabels():
                txt = t.get_text()
                if 'Max Shadow' in txt:
                    t.set_color('red')
                    if include_rejected is False:
                        ax.plot(y[-1], np.arange(len(x_names))[-1], marker='*', color='red')
                    else:
                        idx = 1 + len(rejected_indices)
                        ax.plot(y[-idx], np.arange(len(x_names))[-idx], marker='*', color='red')

            ax.set_ylim((np.arange(len(x_names))[0]-0.5, np.arange(len(x_names))[-1]+0.5))
            #ax.set_xlim((np.min(y)-1, np.max(y)+1))
            ax.invert_yaxis(); ax.invert_xaxis()
        else:
            lns, = ax.plot(np.arange(len(x_names)), y, 'k*--', lw=0.77)#, label='XGBoost', lw=0.77)
            lns_sigma = ax.fill_between(np.arange(len(x_names)), y-y_err, y+y_err, color="grey", alpha=0.2)
            ax.set_ylabel('Z Score', alpha=1, color='k'); ax.set_xticks(np.arange(len(x_names)), x_names, rotation=90)
            for t in ax.get_xticklabels():
                txt = t.get_text()
                if 'Max Shadow' in txt:
                    t.set_color('red')
                    ax.plot(np.arange(len(x_names))[-1], y[-1], marker='*', color='red')
            ax.set_xlim((np.arange(len(x_names))[0]-0.5, np.arange(len(x_names))[-1]+0.5))
            #ax.set_ylim((np.min(y)-1, np.max(y)+1))

        ax.legend([(lns, lns_sigma)], [r'$\pm$ 1$\sigma$'], loc='upper right', ncol=1, frameon=True, fancybox=True, handlelength=2)
        ax.set_title(title)

        plt.tight_layout()

        if savefig:
            plt.savefig('Feature_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.close()#; plt.style.use('default')
        else:
            plt.show()

        return

    def plot_hyper_param_importance(self, plot_time=True, savefig=False):
        """
        Plots hyperparameter importance and, optionally, duration importance.

        Parameters
        ----------
        plot_time : bool
            Include the impact on optimization duration. Defaults to True.
        savefig : bool
            Save a PNG instead of showing. Defaults to False.

        Returns
        -------
        AxesImage
        """

        try:
            if isinstance(self.path, str):
                try:
                    hyper_importances = joblib.load(self.path+'Hyperparameter_Importance')
                except FileNotFoundError:
                    raise ValueError('Could not find the importance file in the '+self.path+' folder')

                try:
                    duration_importances = joblib.load(self.path+'Duration_Importance')
                except FileNotFoundError:
                    raise ValueError('Could not find the importance file in the '+self.path+' folder')
            else:
                raise ValueError('Call the save_hyper_importance() attribute first.')
        except:
            raise ValueError('Call the save_hyper_importance() attribute first.')

        params, importance, duration_importance = [], [], []
        for key in hyper_importances:       
            params.append(key)

        for name in params:
            importance.append(hyper_importances[name])
            duration_importance.append(duration_importances[name])

        xtick_labels = format_labels(params)

        fig, ax = plt.subplots()
        ax.barh(xtick_labels, importance, label='Importance for Classification', color=mcolors.TABLEAU_COLORS["tab:blue"], alpha=0.87)
        if plot_time:
            ax.barh(xtick_labels, duration_importance, label='Impact on Engine Speed', color=mcolors.TABLEAU_COLORS["tab:orange"], alpha=0.7, hatch='/')

        ax.set_ylabel("Hyperparameter"); ax.set_xlabel("Importance Evaluation")
        ax.legend(ncol=2, frameon=False, handlelength=2, bbox_to_anchor=(0.5, 1.1), loc='upper center')
        ax.set_xscale('log'); plt.xlim((0, 1.))
        plt.gca().invert_yaxis()

        if savefig:
            _set_style_()
            if plot_time:
                plt.savefig('Ensemble_Hyperparameter_Importance.png', bbox_inches='tight', dpi=300)
            else:
                plt.savefig('Ensemble_Hyperparameter_Duration_Importance.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

        return

    def save_hyper_importance(self):
        """
        Computes and saves dictionaries of hyperparameter importance and duration
        importance for later plotting.

        Notes
        -----
        Writes two files into the model directory: `Hyperparameter_Importance`
        and `Duration_Importance`. This step can be time-consuming.

        Returns
        -------
        None
        """
        
        print('Calculating and saving importances, this could take up to an hour...')

        try:
            path = self.path if isinstance(self.path, str) else str(Path.home())
        except:
            path = str(Path.home())

        hyper_importance = get_param_importances(self.optimization_results)
        joblib.dump(hyper_importance, path+'Hyperparameter_Importance')

        importance = FanovaImportanceEvaluator()
        duration_importance = importance.evaluate(self.optimization_results, target=lambda t: t.duration.total_seconds())
        joblib.dump(duration_importance, path+'Duration_Importance')
        
        print(f"Files saved in: {path}")

        self.path = path

        return  

#Helper functions below to generate confusion matrix
def format_labels(labels: list) -> list:
    """
    Format hyperparameter/feature labels for display.

    Replaces underscores with spaces, title-cases words, and applies a few
    readable-friendly aliases.

    Parameters
    ----------
    labels : list of str
        Raw label strings to format.

    Returns
    -------
    list of str
        Reformatted labels, same length as the input.
    """
    new_labels = []
    for label in labels:
        label = label.replace("_", " ")
        if label == "eta":
            new_labels.append("Learning Rate"); continue
        if label == "n estimators":
            new_labels.append("Num of Trees"); continue
        if label == "colsample bytree":
            new_labels.append("ColSample ByTree"); continue
        new_labels.append(label.title())

    return new_labels

def evaluate_model(
    classifier, 
    data_x, 
    data_y, 
    normalize=True, 
    k_fold=10, 
    random_state=1909
    ):
    """
    Cross-validates a classifier and returns out-of-fold predictions together with the
    corresponding ground-truth labels.

    Parameters
    ----------
    classifier : estimator
        Any scikit-learn–compatible model implementing `fit` and `predict`.
    data_x : ndarray of shape (n_samples, n_features)
        Feature matrix.
    data_y : array-like of shape (n_samples,)
        Target labels.
    normalize : bool, optional
        Unused in this function; retained for API compatibility with plotting utilities.
        Defaults to True.
    k_fold : int, optional
        Number of K-fold splits. Defaults to 10.
    random_state : int, optional
        Seed for shuffling within the cross-validation splitter. Defaults to 1909.

    Returns
    -------
    predicted_targets : ndarray of shape (n_samples,)
        Out-of-fold predicted labels concatenated across folds.
    actual_targets : ndarray of shape (n_samples,)
        True labels ordered identically to `predicted_targets`.
    """

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    predicted_targets = []
    actual_targets = []

    for train_index, test_index in kf.split(data_x):
        classifier.fit(data_x[train_index], data_y[train_index])
        predicted_targets.extend(classifier.predict(data_x[test_index]))
        actual_targets.extend(data_y[test_index])

    predicted_targets = np.array(predicted_targets)
    actual_targets = np.array(actual_targets)

    return predicted_targets, actual_targets

def generate_matrix(
    predicted_labels_list, 
    actual_targets, 
    classes, 
    normalize=True, 
    title='Confusion Matrix', 
    savefig=False
    ):
    """
    Generate and render a confusion matrix from predicted and true labels.

    Parameters
    ----------
    predicted_labels_list : array-like of shape (n_samples,)
        Predicted class labels, typically the out-of-fold predictions returned by `evaluate_model()`.
    actual_targets : array-like of shape (n_samples,)
        Ground-truth class labels in the same order as `predicted_labels_list`.
    classes : list of str
        Class names used to label the matrix axes. The order must match the label encoding in the inputs.
    normalize : bool, optional
        If True the confusion matrix is normalized (row-wise) before plotting. Defaults to True.
    title : str, optional
        Figure title. Defaults to 'Confusion Matrix'.
    savefig : bool, optional
        If True the figure is saved to 'Ensemble_Confusion_Matrix.png' and not displayed. Defaults to False.

    Returns
    -------
    None
        Displays the figure or saves it to disk.
    """

    conf_matrix = confusion_matrix(actual_targets, predicted_labels_list)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(8,8))
    if normalize:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title=title, savefig=savefig)
    else:
        generate_plot(conf_matrix, classes=classes, normalize=normalize, title=title, savefig=savefig)
    
    if savefig:
        plt.savefig('Ensemble_Confusion_Matrix.png', bbox_inches='tight', dpi=300)
        plt.clf()
    else:
        plt.show()
    
def generate_plot(
    conf_matrix, 
    classes, 
    normalize=False, 
    title='Confusion Matrix', 
    include_cbar=False, 
    savefig=False
    ):
    """
    Generate a confusion-matrix figure and axes without calling `plt.show()`.

    Parameters
    ----------
    conf_matrix : array-like of shape (n_classes, n_classes)
        Confusion matrix (counts) produced upstream (e.g., via `confusion_matrix`).
    classes : list of str
        Class names used for tick labels. Order must match the matrix axes.
    normalize : bool, optional
        If True the matrix is normalized row-wise to proportions. Defaults to False.
    title : str, optional
        Figure title. Defaults to 'Confusion Matrix'.
    include_cbar : bool, optional
        If True a colorbar is added to the figure. Defaults to False.
    savefig : bool, optional
        Included for API symmetry; saving is typically handled by the caller. Defaults to False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the confusion matrix.
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8,8)) 
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    
    # Adjust the colorbar to match the matrix height
    if include_cbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend='both')

    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, alpha=1, color='k')
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, alpha=1, color='k', rotation=90)

    fmt = '.4f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.

    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        ax.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

    ax.set_ylabel('True label', alpha=1, color='k')
    ax.set_xlabel('Predicted label', alpha=1, color='k')
    ax.grid(False)
    fig.tight_layout()

    return fig, ax
    
def _set_style_():
    """
    Function to configure the matplotlib.pyplot style. This function is called before any images are saved,
    after which the style is reset to the default.
    """

    plt.rcParams["xtick.color"] = "323034"
    plt.rcParams["ytick.color"] = "323034"
    plt.rcParams["text.color"] = "323034"
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["patch.facecolor"] = "#bc80bd"  # Replace with a valid color code
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["patch.linewidth"] = 0.8
    plt.rcParams["scatter.edgecolors"] = "black"
    plt.rcParams["grid.color"] = "#b1afb5"  # Replace with a valid color code
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.title_fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.prop_cycle"] = (cycler('color', ['#bc80bd', '#fb8072', '#b3de69', '#fdb462', '#fccde5', '#8dd3c7', '#ffed6f', '#bebada', '#80b1d3', '#ccebc5', '#d9d9d9']))  # Replace with valid color codes
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.8
    plt.rcParams["legend.fontsize"] = 13
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["legend.borderpad"] = 0.2
    plt.rcParams["legend.columnspacing"] = 1.5
    plt.rcParams["legend.labelspacing"] = 0.4
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.labelsize"] = 17
    plt.rcParams["axes.titlelocation"] = "center"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["axes.autolimit_mode"] = "round_numbers"
    plt.rcParams["axes.labelpad"] = 3
    plt.rcParams["axes.formatter.limits"] = (-4, 4)
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.top"] = True
    plt.rcParams["figure.titlesize"] = 18
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300

    return

