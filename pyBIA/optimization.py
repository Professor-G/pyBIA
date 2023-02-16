#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  11 12:04:23 2021

@author: daniel
"""
import os, sys
import tensorflow as tf
os.environ['PYTHONHASHSEED'], os.environ["TF_DETERMINISTIC_OPS"] = '0', '1'
import numpy as np
import random as python_random
##https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development##
np.random.seed(1909), python_random.seed(1909), tf.random.set_seed(1909)
from pandas import DataFrame
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
from collections import Counter 

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score

from missingpy import MissForest
from skopt import BayesSearchCV, plots, gp_minimize
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical 
from tensorflow.keras.backend import clear_session 
from tensorflow.keras.callbacks import EarlyStopping, Callback

from boruta import BorutaPy
from BorutaShap import BorutaShap
from xgboost import XGBClassifier, DMatrix, train

import optuna
from optuna.integration import TFKerasPruningCallback
optuna.logging.set_verbosity(optuna.logging.WARNING)
from pyBIA.data_augmentation import augmentation, resize
from pyBIA import data_processing, cnn_model

class objective_cnn(object):
    """
    Optimization objective function for pyBIA's convolutional neural network. 
    This is passed through the hyper_opt() function when optimizing with
    Optuna. The Optuna software for hyperparameter optimization was published in 
    2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    Unlike the objective functions for the ensemble algorithms, this takes as input
    the two classes directly, instead of the traditional data (data_x) and accompanying
    label array (data_y). This is because the cnn_model.AlexNet() function takes as input the
    two classes, after which it automatically assigns the 0 and 1 label, respectively. 
    The training parameters are optimized even if opt_aug is True.

    Note:
        If opt_aug is enabled, then the class1 sample will be the data that will augmented.
        It is best to keep both class sizes the same after augmentation, therefore balance=True
        by default, which will truncate the class2 sample to meet the augmented class1 size.

        For example, if class1 contains only 100 images, opt_aug will identify the ideal number 
        of augmentations to perform. If this ideal quantity is 10, then each of the 100 class1 images will 
        be augmented 10 times, and thus only the first 1000 images in the class2 sample will be used so
        as to keep the final sample sizes the same. 

        Since the maximum number of augmentations allowed is batch_max per each sample, in practice class2 
        should contain batch_max times the size of class1. During the optimization procedure, if an 
        augmentation batch size of 200 is assesed, then 100*200=20000 augmented images will be created, 
        and therefore during that particular trial 20000 images from class2 will be used, and so forth. 
        If class2 does not contain 20000 samples, then all will be used.

        To use the entire class2 sample regardless of the number augmentations performed, set balance_val=False.
    
        The min_pixel and max_pixel value will be used to normalize the images if Normalize=True. The opt_max_min_pix
        and opt_max_max_pix, when set, will be used instead during the optimization. If optimizing the normalization
        scheme the default min_pixel will be set to zero, as such only the optimal max_pixel for every band will be output.

    Args:
        class1
        class2
        img_num_channels
    """

    def __init__(self, class1, class2, img_num_channels=1, normalize=True, min_pixel=0,
        max_pixel=100, val_blob=None, val_other=None, train_epochs=25, patience=20, 
        opt_model=True, opt_aug=False, batch_min=10, batch_max=250, image_size_min=50, 
        image_size_max=100, balance_val=True, opt_max_min_pix=None, opt_max_max_pix=None, 
        metric='loss', average=True):

        self.class1 = class1
        self.class2 = class2
        self.img_num_channels = img_num_channels
        self.normalize = normalize 
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.val_blob = val_blob
        self.val_other = val_other
        self.train_epochs = train_epochs
        self.patience = patience 
        self.opt_model = opt_model  
        self.opt_aug = opt_aug
        self.batch_min = batch_min 
        self.batch_max = batch_max 
        self.image_size_min = image_size_min
        self.image_size_max = image_size_max
        self.balance_val = balance_val
        self.opt_max_min_pix = opt_max_min_pix
        self.opt_max_max_pix = opt_max_max_pix
        self.metric = metric 
        self.average = average

        #if self.val_blob is None and self.val_other is None and self.average:
        #   print('Cannot average metrics without validation data, setting average=False.')
        #   self.average = False 
        if 'all' not in self.metric and 'loss' not in self.metric and 'f1_score' not in self.metric and 'binary_accuracy' not in self.metric:
            raise ValueError("Invalid metric input, options are: 'loss', 'binary_accuracy', 'f1_score', or 'all', and the validation equivalents (add val_ at the beginning).")
        
        if self.metric == 'val_loss' or self.metric == 'val_binary_accuracy':
            if self.val_blob is None and self.val_other is None:
                raise ValueError('No validation data input, change the metric to either "loss" or "binary_accuracy".')

        if self.opt_max_min_pix is not None:
            if self.opt_max_max_pix is None:
                raise ValueError('To optimize min/max normalization pixel value, both opt_min_pix and opt_max_pix must be input')

        if self.opt_max_max_pix is not None:
            if self.opt_max_min_pix is None:
                raise ValueError('To optimize min/max normalization pixel value, both opt_min_pix and opt_max_pix must be input')

    def __call__(self, trial):

        if self.opt_aug:
            if self.img_num_channels == 1:
                channel1, channel2, channel3 = self.class1, None, None 
            elif self.img_num_channels == 2:
                channel1, channel2, channel3 = self.class1[:,:,:,0], self.class1[:,:,:,1], None 
            elif self.img_num_channels == 3:
                channel1, channel2, channel3 = self.class1[:,:,:,0], self.class1[:,:,:,1], self.class1[:,:,:,2]
            else:
                raise ValueError('Only three filters are supported!')

        clear_session()

        if 'loss' in self.metric:
            mode = 'min'
        else:
            mode = 'max'

        if self.opt_aug:
            batch = trial.suggest_int('batch', self.batch_min, self.batch_max, step=1)
            image_size = trial.suggest_int('image_size', self.image_size_min, self.image_size_max, step=1)
            shift = 10 #trial.suggest_int('shift', 0, 10)
            horizontal = vertical = rotation = True #trial.suggest_categorical('horizontal', [True, False])
      
            augmented_images = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=batch, 
                width_shift=shift, height_shift=shift, horizontal=horizontal, vertical=vertical, rotation=rotation, 
                image_size=image_size)

            if self.img_num_channels > 1:
                class_1=[]
                if self.img_num_channels == 2:
                    for i in range(len(augmented_images[0])):
                        class_1.append(data_processing.concat_channels(augmented_images[0][i], augmented_images[1][i]))
                else:
                    for i in range(len(augmented_images[0])):
                        class_1.append(data_processing.concat_channels(augmented_images[0][i], augmented_images[1][i], augmented_images[2][i]))
                class_1 = np.array(class_1)
            else:
                class_1 = augmented_images

            if self.balance_val:
                class_2 = self.class2[:len(class_1)]   
            else:
                class_2 = self.class2   

            if self.img_num_channels == 1:
                class_2 = resize(class_2, size=image_size)
            else:
                channel1 = resize(class_2[:,:,:,0], size=image_size)
                channel2 = resize(class_2[:,:,:,1], size=image_size)
                if self.img_num_channels == 2:
                    class_2 = data_processing.concat_channels(channel1, channel2)
                else:
                    channel3 = resize(class_2[:,:,:,2], size=image_size)
                    class_2 = data_processing.concat_channels(channel1, channel2, channel3)

            #Need to also crop the validation images
            if self.val_blob is not None:
                if self.img_num_channels == 1:
                    val_class_1 = resize(self.val_blob, size=image_size)
                else:
                    val_channel1 = resize(self.val_blob[:,:,:,0], size=image_size)
                    val_channel2 = resize(self.val_blob[:,:,:,1], size=image_size)
                    if self.img_num_channels == 2:
                        val_class_1 = data_processing.concat_channels(val_channel1, val_channel2)
                    else:
                        val_channel3 = resize(self.val_blob[:,:,:,2], size=image_size)
                        val_class_1 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)
            else:
                val_class_1 = None 

            if self.val_other is not None:
                if self.img_num_channels == 1:
                    val_class_2 = resize(self.val_other, size=image_size)
                elif self.img_num_channels > 1:
                    val_channel1 = resize(self.val_other[:,:,:,0], size=image_size)
                    val_channel2 = resize(self.val_other[:,:,:,1], size=image_size)
                    if self.img_num_channels == 2:
                        val_class_2 = data_processing.concat_channels(val_channel1, val_channel2)
                    else:
                        val_channel3 = resize(self.val_other[:,:,:,2], size=image_size)
                        val_class_2 = data_processing.concat_channels(val_channel1, val_channel2, val_channel3)
            else:
                val_class_2 = None 

        else:
            class_1, class_2 = self.class1, self.class2
            val_class_1, val_class_2 = self.val_blob, self.val_other

        if self.opt_max_min_pix is not None:
            self.normalize = True #Just in case it's set to False by the user 
            min_pix, max_pix = 0.0, []
            if self.img_num_channels >= 1:
                max_pix_1 = trial.suggest_int('max_pixel_1', self.opt_max_min_pix, self.opt_max_max_pix, step=5)
                max_pix.append(max_pix_1)
            if self.img_num_channels >= 2:
                max_pix_2 = trial.suggest_int('max_pixel_2', self.opt_max_min_pix, self.opt_max_max_pix, step=5)
                max_pix.append(max_pix_2)
            if self.img_num_channels == 3:
                max_pix_3 = trial.suggest_int('max_pixel_3', self.opt_max_min_pix, self.opt_max_max_pix, step=5)
                max_pix.append(max_pix_3)
            elif self.img_num_channels > 3:
                raise ValueError('Only up to three channels are currently supported!')
        else:
            min_pix, max_pix = self.min_pixel, self.max_pixel

        if self.patience != 0:
            if self.metric == 'all' or self.metric == 'val_all':
                print('Cannot use callbacks if averaging out all performance metrics for evaluation, setting patience=0.')
                callbacks = None
            else:
                callbacks = [EarlyStopping(monitor=self.metric, mode=mode, patience=self.patience), TFKerasPruningCallback(trial, monitor=self.metric),]
        else:
            callbacks = None

        ### Training parameters ###
        lr = trial.suggest_float('lr', 1e-6, 0.01, step=0.05)
        decay = trial.suggest_float('decay', 0, 0.1, step=0.001)
        batch_size = trial.suggest_int('batch_size', 16, 64)
        momentum = trial.suggest_float('momentum', 0.0, 1.0, step=0.05)
        nesterov = trial.suggest_categorical('nesterov', [True, False])

        if self.opt_model:
            """AlexNet hyperparameter search space"""

            ### Activation and Loss Functions ### 
            activation_conv = trial.suggest_categorical('activation_conv', ['relu',  'sigmoid', 'tanh'])            
            activation_dense = trial.suggest_categorical('activation_dense', ['relu', 'sigmoid', 'tanh'])
            regularizer = trial.suggest_categorical('regularizer', ['local_response', 'batch_norm'])
            loss = trial.suggest_categorical('loss', ['binary_crossentropy', 'squared_hinge'])

            ### Filter and layer hyperparameters ###
            filter_1 = trial.suggest_int('filter_1', 12, 408, step=12)
            filter_size_1 = trial.suggest_int('filter_size_1', 1, 7, step=2)
            strides_1 = trial.suggest_int('strides_1', 1, 3)
            pooling_1 = trial.suggest_categorical('pooling_1', ['max', 'average'])
            pool_size_1 = trial.suggest_int('pool_size_1', 2, 7)
            pool_stride_1 = trial.suggest_int('pool_stride_1', 1, 3)

            filter_2 = trial.suggest_int('filter_2', 12, 408, step=12)
            filter_size_2 = trial.suggest_int('filter_size_2', 1, 7, step=2)
            strides_2 = trial.suggest_int('strides_2', 1, 3)
            pooling_2 = trial.suggest_categorical('pooling_2', ['max', 'average'])
            pool_size_2 = trial.suggest_int('pool_size_2', 2, 7)
            pool_stride_2 = trial.suggest_int('pool_stride_2', 1, 3)

            filter_3 = trial.suggest_int('filter_3', 12, 408, step=12)
            filter_size_3 = trial.suggest_int('filter_size_3', 1, 7, step=2)
            strides_3 = trial.suggest_int('strides_3', 1, 3)
            pooling_3 = trial.suggest_categorical('pooling_3', ['max', 'average'])
            pool_size_3 = trial.suggest_int('pool_size_3', 2, 7)
            pool_stride_3 = trial.suggest_int('pool_stride_3', 1, 3)

            filter_4 = trial.suggest_int('filter_4', 12, 408, step=12)
            filter_size_4 = trial.suggest_int('filter_size_4', 1, 7, step=2)
            strides_4 = trial.suggest_int('strides_4', 1, 3)

            filter_5 = trial.suggest_int('filter_5', 12, 408, step=12)
            filter_size_5 = trial.suggest_int('filter_size_5', 1, 7, step=2)
            strides_5 = trial.suggest_int('strides_5', 1, 3) 

            ### Dense Layers ###
            dense_neurons_1 = trial.suggest_int('dense_neurons_1', 128, 6400, step=128)
            dense_neurons_2 = trial.suggest_int('dense_neurons_2', 128, 6400, step=128)
            dropout_1 = trial.suggest_float('dropout_1', 0, 0.5, step=0.05)
            dropout_2 = trial.suggest_float('dropout_2', 0, 0.5, step=0.05) 

            #try:
            model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, 
                normalize=self.normalize, min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, 
                epochs=self.train_epochs, batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, 
                loss=loss, activation_conv=activation_conv, activation_dense=activation_dense, regularizer=regularizer, 
                pooling_1=pooling_1, pooling_2=pooling_2, pooling_3=pooling_3, pool_size_1=pool_size_1, 
                pool_stride_1=pool_stride_1, pool_size_2=pool_size_2, pool_stride_2=pool_stride_2, 
                pool_size_3=pool_size_3, pool_stride_3=pool_stride_3, filter_1=filter_1, filter_size_1=filter_size_1, 
                strides_1=strides_1, filter_2=filter_2, filter_size_2=filter_size_2, strides_2=strides_2, 
                filter_3=filter_3, filter_size_3=filter_size_3, strides_3=strides_3, filter_4=filter_4, 
                filter_size_4=filter_size_4, strides_4=strides_4, filter_5=filter_5, filter_size_5=filter_size_5, 
                strides_5=strides_5, dense_neurons_1=dense_neurons_1, dense_neurons_2=dense_neurons_2, 
                dropout_1=dropout_1, dropout_2=dropout_2, early_stop_callback=callbacks, checkpoint=False)    
            #except:
                #print("Invalid hyperparameter combination, skipping trial.")
                #return 0.0
        else:
            model, history = cnn_model.AlexNet(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.train_epochs, 
                batch_size=batch_size, lr=lr, momentum=momentum, decay=decay, nesterov=nesterov, early_stop_callback=callbacks, checkpoint=False)

        metrics = ['loss', 'binary_accuracy', 'f1_score']
        if self.metric == 'all': #Average all the training metrics
            if self.average:
                training_metrics_mean = np.mean([np.mean(history.history[metric]) for metric in metrics if 'loss' not in metric])
                training_loss_mean = 1 - np.mean(history.history['loss']) # negative sign because loss is being minimized
            else:
                training_metrics_mean = np.mean([history.history[metric][-1] for metric in metrics if 'loss' not in metric])
                training_loss_mean = 1 - history.history['loss'][-1] 
            final_score = np.mean([training_metrics_mean, training_loss_mean])
        elif self.metric == 'val_all': #Average all the validation metrics
            if self.average:
                val_metrics_mean = np.mean([np.mean(history.history['val_'+metric]) for metric in metrics if 'loss' not in metric])
                val_loss_mean = 1 - np.mean(history.history['val_loss']) 
            else:
                val_metrics_mean = np.mean([history.history['val_'+metric][-1] for metric in metrics if 'loss' not in metric])
                val_loss_mean = 1 - history.history['val_loss'][-1]
            final_score = np.mean([val_metrics_mean, val_loss_mean])
        else:
            if self.average:
                final_score = np.mean(history.history[self.metric])
            else:
                final_score = history.history[self.metric][-1]
            if 'loss' in self.metric: 
                final_score = 1 - final_score

        return final_score

class objective_xgb(object):
    """
    Optimization objective function for the tree-based XGBoost classifier. 
    The Optuna software for hyperparameter optimization was published in 
    2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902

    If test_cv is None CV is used (stratified kfold) and no pruning is used during
    the training  
    """

    def __init__(self, data_x, data_y, limit_search=False, opt_cv=3, test_size=None):
        self.data_x = data_x
        self.data_y = data_y
        self.limit_search = limit_search
        self.opt_cv = opt_cv 
        self.test_size = test_size 

    def __call__(self, trial):

        params = {"objective": "binary:logistic", "eval_metric": "auc"}
    
        if self.test_size is not None:
            train_x, valid_x, train_y, valid_y = train_test_split(self.data_x, self.data_y, test_size=self.test_size, random_state=np.random.randint(1, 1e9))
            dtrain, dvalid = DMatrix(train_x, label=train_y), DMatrix(valid_x, label=valid_y)
            #print('Initializing XGBoost Pruner...')
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")

        if self.limit_search:
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 250)
            params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
            params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-8, 1)
            params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-8, 1)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 25)
            params['eta'] = trial.suggest_loguniform('eta', 1e-8, 1)
            params['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1)
            params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

            if params['booster'] == "dart":
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1)
                params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1)
                if self.test_size is None:
                    clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                        reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                        grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],
                        rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], random_state=1909)#, tree_method='hist')
            
            elif params['booster'] == 'gbtree':
                params['subsample'] = trial.suggest_loguniform('subsample', 1e-6, 1.0)
                if self.test_size is None:
                    clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                        reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                        grow_policy=params['grow_policy'], subsample=params['subsample'], random_state=1909)#, tree_method='hist')

            if self.test_size is not None:
                bst = train(params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
                preds = bst.predict(dvalid)
                pred_labels = np.rint(preds)
                accuracy = accuracy_score(valid_y, pred_labels)
            else:
                #FROM SKLEARN DOCUMENTATION: For int/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, Fold is used.
                cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv) 
                accuracy = np.mean(cv['test_score'])

            return accuracy

        params['booster'] = trial.suggest_categorical('booster', ['gbtree', 'dart'])
        params['n_estimators'] = trial.suggest_int('n_estimators', 100, 500)
        params['reg_lambda'] = trial.suggest_float('reg_lambda', 0, 100)
        params['reg_alpha'] = trial.suggest_int('reg_alpha', 0, 100)
        params['max_depth'] = trial.suggest_int('max_depth', 2, 25)
        params['eta'] = trial.suggest_float('eta', 1e-8, 1)
        params['gamma'] = trial.suggest_int('gamma', 1, 100)
        params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 100)
        params['max_delta_step'] = trial.suggest_int('max_delta_step', 1, 100)
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
        params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1)

        if params['booster'] == "dart":
            params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1)
            params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1)
            if self.test_size is None:
                clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], 
                    gamma=params['gamma'], grow_policy=params['grow_policy'], min_child_weight=params['min_child_weight'], 
                    max_delta_step=params['max_delta_step'], subsample=params['subsample'], sample_type=params['sample_type'], 
                    normalize_type=params['normalize_type'], rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], random_state=1909)#, tree_method='hist')
        elif params['booster'] == 'gbtree':
            if self.test_size is None:
                clf = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'],  reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], grow_policy=params['grow_policy'], 
                    min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'], subsample=params['subsample'], random_state=1909)#, tree_method='hist')
            
        if self.test_size is not None:
            bst = train(params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            accuracy = accuracy_score(valid_y, pred_labels)
        else:
            cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
            accuracy = np.mean(cv['test_score'])
        
        return accuracy


class objective_nn(object):
    """
    Optimization objective function for the scikit-learn implementatin of the
    MLP classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902
    """

    def __init__(self, data_x, data_y, opt_cv):
        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv

    def __call__(self, trial):
        learning_rate_init= trial.suggest_float('learning_rate_init', 1e-5, 0.1, step=0.05)
        solver = trial.suggest_categorical("solver", ["sgd", "adam"]) #"lbfgs"
        activation = trial.suggest_categorical("activation", ["logistic", "tanh", "relu"])
        learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        alpha = trial.suggest_float("alpha", 1e-6, 1, step=0.05)
        batch_size = trial.suggest_int('batch_size', 1, 1000)
        
        n_layers = trial.suggest_int('hidden_layer_sizes', 1, 10)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', 100, 5000))

        try:
            clf = MLPClassifier(hidden_layer_sizes=tuple(layers),learning_rate_init=learning_rate_init, 
                solver=solver, activation=activation, alpha=alpha, batch_size=batch_size, max_iter=2500, random_state=1909)
        except:
            print("Invalid hyperparameter combination, skipping trial")
            return 0.0

        cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
        final_score = np.mean(cv['test_score'])

        return final_score

class objective_rf(object):
    """
    Optimization objective function for the scikit-learn implementatin of the
    Random Forest classifier. The Optuna software for hyperparameter optimization
    was published in 2019 by Akiba et al. Paper: https://arxiv.org/abs/1907.10902
    """

    def __init__(self, data_x, data_y, opt_cv):
        self.data_x = data_x
        self.data_y = data_y
        self.opt_cv = opt_cv

    def __call__(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 3000)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        max_depth = trial.suggest_int('max_depth', 2, 25)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 25)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
        max_features = trial.suggest_int('max_features', 1, self.data_x.shape[1])
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        try:
            clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                max_features=max_features, bootstrap=bootstrap, random_state=1909)
        except:
            print("Invalid hyperparameter combination, skipping trial")
            return 0.0

        cv = cross_validate(clf, self.data_x, self.data_y, cv=self.opt_cv)
        final_score = np.mean(cv['test_score'])

        return final_score


def hyper_opt(data_x, data_y, clf='rf', n_iter=25, return_study=True, balance=True, img_num_channels=1, 
    normalize=True, min_pixel=0, max_pixel=100, val_X=None, val_Y=None, train_epochs=25, patience=5, metric='loss', 
    limit_search=True, opt_model=True, opt_aug=False, batch_min=10, batch_max=300, image_size_min=50, image_size_max=100, balance_val=True,
    opt_max_min_pix=None, opt_max_max_pix=None, opt_cv=10, test_size=None, average=True):
    """
    Optimizes hyperparameters using a k-fold cross validation splitting strategy, unless a CNN
    is being optimized, in which case no cross-validation is performed during trial assesment.

    **IMPORTANT** In the case of CNN optimization, data_x and data_y are not the standard
    data plus labels -- if optimizing a CNN the samples for the first class should be passed
    through the data_x parameter, and the samples for the second class should be given as data_y.
    These two classes will automatically be assigned the label 0 and 1, respectively.

    Likewise, if optimizing a CNN model, val_X corresponds to the images of the first class, 
    and val_Y the images of the second class. These will be automatically processed
    with class labels of 0 and 1, respectively.
    
    Note:
        If save_study=True, the Optuna study object will be the third output. This
        object can be used for various analysis, including optimization visualization.

        See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html

    Example:
        The function will create the classification engine and optimize the hyperparameters
        using an iterative approach:

        >>> model, params = hyper_opt(data_x, data_y, clf='rf') 
        
        The first output is our optimal classifier, and will be used to make predictions:
        
        >>> prediction = model.predict(new_data)
        
        The second output of the optimize function is the dictionary containing
        the hyperparameter combination that yielded the highest mean accuracy.

        If save_study = True, the Optuna study object will also be returned as the third output.
        This can be used to plot the optimization results, see: https://optuna.readthedocs.io/en/latest/tutorial/10_key_features/005_visualization.html#sphx-glr-tutorial-10-key-features-005-visualization-py

        >>> from optuna.visualization.matplotlib import plot_contour
        >>> 
        >>> model, params, study = hyper_opt(data_x, data_y, clf='rf', save_study=True) 
        >>> plot_contour(study)
        
    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features. In the case of 
            CNN, the samples for the first class should be passed, which will
            automatically be assigned the label '0'.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        clf (str): The machine learning classifier to optimize. Can either be
            'rf' for Random Forest, 'nn' for Neural Network, 'xgb' for eXtreme Gradient Boosting,
            or 'cnn' for Convolutional Neural Network. Defaults to 'rf'.
            In the case of CNN, the samples for the second class should be passed, which will
            automatically be assigned the label '1'.
        n_iter (int, optional): The maximum number of iterations to perform during 
            the hyperparameter search. Defaults to 25.
        return_study (bool, optional): If True the Optuna study object will be returned. This
            can be used to review the method attributes, such as optimization plots. Defaults to True.
        balance (bool, optional): If True, a weights array will be calculated and used
            when fitting the classifier. This can improve classification when classes
            are imbalanced. This is only applied if the classification is a binary task. 
            Defaults to True. Argument is ignored if clf='cnn'.
        img_num_channels (int): The number of filters used. Defaults to 1, as pyBIA version 1
            has been trained with only blue broadband data. Only used when clf = 'cnn'.
        normalize (bool, optional): If True the data will be min-max normalized using the 
            input min and max pixels. Defaults to True. Only used when clf = 'cnn'.
        min_pixel (int, optional): The minimum pixel count, defaults to 0. Pixels with counts 
            below this threshold will be set to this limit. Only used when clf = 'cnn'.
        max_pixel (int, optional): The maximum pixel count, defaults to 100. Pixels with counts 
            above this threshold will be set to this limit. Only used when clf = 'cnn'.
        val_X (array, optional): 3D matrix containing the 2D arrays (images)
            to be used for validation.
        val_Y (array, optional): A binary class matrix containing the labels of the
            corresponding validation data. This binary matrix representation can be created
            using tensorflow, see example in the Notes.
        train_epochs (int): Number of epochs used for training. The model accuracy will be
            the validation accuracy at the end of this epoch. 
        patience (int): Number of epochs without improvement before  the optimization trial
            is terminated. 
        metric (str): Assesment metric to use when both pruning and scoring the hyperparameter
            optimization trial.
        limit_search (bool): If True the XGB optimization search space will be limited,
            for computational and time purposes. Defaults to True.
        opt_cv (int): Cross-validations to perform when assesing the performance at each
            hyperparameter optimization trial. For example, if cv=3, then each optimization trial
            will be assessed according to the 3-fold cross validation accuracy. Defaults to 10.
            NOTE: The higher this number, the longer the optimization will take.
            
    Returns:
        The first output is the classifier with the optimal hyperparameters.
        Second output is a dictionary containing the optimal hyperparameters.
        If save_study=True, the Optuna study object will be the third output.
    """

    if clf == 'rf':
        model_0 = RandomForestClassifier(random_state=1909)
    elif clf == 'nn':
        model_0 = MLPClassifier(random_state=1909)
    elif clf == 'xgb':
        model_0 = XGBClassifier(random_state=1909)
        if all(isinstance(val, (int, str)) for val in data_y):
            print('XGBoost classifier requires numerical class labels! Converting class labels as follows:')
            print('____________________________________')
            y = np.zeros(len(data_y))
            for i in range(len(np.unique(data_y))):
                print(str(np.unique(data_y)[i]).ljust(10)+'  ------------->     '+str(i))
                index = np.where(data_y == np.unique(data_y)[i])[0]
                y[index] = i
            data_y = y 
            print('------------------------------------')
    elif clf == 'cnn':
        pass 
    else:
        raise ValueError('clf argument must either be "rf", "xgb", "nn", or "cnn".')

    if clf != 'cnn':
        cv = cross_validate(model_0, data_x, data_y, cv=opt_cv)
        initial_score = np.mean(cv['test_score'])

    sampler = optuna.samplers.TPESampler(seed=1909)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
    print('Starting hyperparameter optimization, this will take a while...')
    #If binary classification task, can deal with imbalance classes with weights hyperparameter
    if len(np.unique(data_y)) == 2 and clf != 'cnn':
        counter = Counter(data_y)
        if counter[np.unique(data_y)[0]] != counter[np.unique(data_y)[1]]:
            if balance:
                print('Unbalanced dataset detected, will train classifier with weights! To disable, set balance=False')
                if clf == 'xgb':
                    total_negative = len(np.where(data_y == counter.most_common(1)[0][0])[0])
                    total_positive = len(data_y) - total_negative
                    sample_weight = total_negative / total_positive
                elif clf == 'rf':
                    sample_weight = np.zeros(len(data_y))
                    for i,label in enumerate(np.unique(data_y)):
                        index = np.where(data_y == label)[0]
                        sample_weight[index] = len(index) 
                elif clf == 'nn':
                    print('WARNING: Unbalanced dataset detected but MLPClassifier() does not support sample weights.')
            else:
                sample_weight = None
        else:
            sample_weight = None

    if clf == 'rf':
        try:
            objective = objective_rf(data_x, data_y, opt_cv=opt_cv)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True, gc_after_trial=True)
            params = study.best_trial.params
            model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], 
                max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], 
                min_samples_leaf=params['min_samples_leaf'], max_features=params['max_features'], 
                bootstrap=params['bootstrap'], class_weight=sample_weight, random_state=1909)
        except:
            print('Failed to optimize with Optuna, switching over to BayesSearchCV...')
            params = {
                'criterion': ["gini", "entropy"],
                'n_estimators': [int(x) for x in np.linspace(50,1000, num=20)], 
                'max_features': [data_x.shape[1], "sqrt", "log2"],
                'max_depth': [int(x) for x in np.linspace(5,50,num=5)],
                'min_samples_split': [3,4,6,7,8,9,10],
                'min_samples_leaf': [1,3,5,7,9,10],
                'max_leaf_nodes': [int(x) for x in np.linspace(2,200)],
                'bootstrap': [True,False]   
            }
            gs = BayesSearchCV(n_iter=n_iter, estimator=RandomForestClassifier(random_state=1909), search_spaces=params, 
                optimizer_kwargs={'base_estimator': 'RF'}, cv=opt_cv)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_

    elif clf == 'nn':
        try:
            objective = objective_nn(data_x, data_y, opt_cv=opt_cv)
            study.optimize(objective, n_trials=n_iter, show_progress_bar=True, gc_after_trial=True)
            params = study.best_trial.params
            layers = [param for param in params if 'n_units_' in param]
            layers = tuple(params[layer] for layer in layers)
            model = MLPClassifier(hidden_layer_sizes=tuple(layers), learning_rate_init=params['learning_rate_init'], 
                activation=params['activation'], learning_rate=params['learning_rate'], alpha=params['alpha'], 
                batch_size=params['batch_size'], solver=params['solver'], max_iter=2500, random_state=1909)
        except:
            print('Failed to optimize with Optuna, switching over to BayesSearchCV...')
            params = {
                'hidden_layer_sizes': [(100,),(50,100,50),(75,50,20),(150,100,50),(120,80,40),(100,50,30)],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'alpha': [0.00001, 0.5], 
                'activation': ['tanh', 'logistic', 'relu'],
                'solver': ['sgd', 'adam'],
                'max_iter': [100, 150, 200] 
            }
            gs = BayesSearchCV(n_iter=n_iter, estimator=MLPClassifier(random_state=1909), search_spaces=params, cv=opt_cv)
            gs.fit(data_x, data_y)
            best_est, best_score = gs.best_estimator_, np.round(gs.best_score_, 4)
            print('Highest mean accuracy: {}'.format(best_score))
            return gs.best_estimator_, gs.best_params_
          
    elif clf == 'xgb':
        objective = objective_xgb(data_x, data_y, limit_search=limit_search, opt_cv=opt_cv, test_size=test_size)
        if limit_search:
            print('NOTE: To expand hyperparameter search space, set limit_search=False, although this will increase the optimization time significantly.')
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True, gc_after_trial=True)#gc_after_trial=True
        params = study.best_trial.params
        if limit_search:
            if params['booster'] == 'dart':
                model = XGBClassifier(booster=params['booster'],  n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],
                    rate_drop=params['rate_drop'], skip_drop=params['skip_drop'], scale_pos_weight=sample_weight, random_state=1909)
            elif params['booster'] == 'gbtree':
                model = XGBClassifier(booster=params['booster'],  n_estimators=params['n_estimators'], reg_lambda=params['reg_lambda'], 
                    reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], subsample=params['subsample'], scale_pos_weight=sample_weight, random_state=1909)
        else:
            if params['booster'] == 'dart':
                model = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], sample_type=params['sample_type'], normalize_type=params['normalize_type'],rate_drop=params['rate_drop'], 
                    skip_drop=params['skip_drop'], min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'], subsample=params['subsample'],
                    scale_pos_weight=sample_weight, random_state=1909)
            elif params['booster'] == 'gbtree':
                model = XGBClassifier(booster=params['booster'], n_estimators=params['n_estimators'], colsample_bytree=params['colsample_bytree'], 
                    reg_lambda=params['reg_lambda'], reg_alpha=params['reg_alpha'], max_depth=params['max_depth'], eta=params['eta'], gamma=params['gamma'], 
                    grow_policy=params['grow_policy'], subsample=params['subsample'], min_child_weight=params['min_child_weight'], max_delta_step=params['max_delta_step'],
                    scale_pos_weight=sample_weight, random_state=1909)
       
    else:
        objective = objective_cnn(data_x, data_y, img_num_channels=img_num_channels, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, 
            val_blob=val_X, val_other=val_Y, train_epochs=train_epochs, patience=patience, metric=metric, average=average, opt_model=opt_model, 
            opt_aug=opt_aug, batch_min=batch_min, batch_max=batch_max, image_size_min=image_size_min, image_size_max=image_size_max, balance_val=balance_val, 
            opt_max_min_pix=opt_max_min_pix, opt_max_max_pix=opt_max_max_pix)
        study.optimize(objective, n_trials=n_iter, show_progress_bar=True, gc_after_trial=True)#, n_jobs=1)
        params = study.best_trial.params

    final_score = study.best_value

    if clf != 'cnn':
        if initial_score > final_score:
            print('Hyperparameter optimization complete! Optimal accuracy of {} is LOWER than the base accuracy of {}, try increasing the value of n_iter and run again.'.format(np.round(final_score, 6), np.round(initial_score, 6)))
        else:
            print('Hyperparameter optimization complete! Optimal accuracy of {} is HIGHER than the base accuracy of {}.'.format(np.round(final_score, 6), np.round(initial_score, 6)))
        if return_study:
            return model, params, study
        return model, params
    else:
        print('Hyperparameter optimization complete! Best validation accuracy: {}'.format(np.round(final_score, 4)))
        if return_study:
            return params, study
        return params

def borutashap_opt(data_x, data_y, boruta_trials=50, model='rf', importance_type='gain'):
    """
    Applies a combination of the Boruta algorithm and
    Shapley values, a method developed by Eoghan Keany (2020).

    See: https://doi.org/10.5281/zenodo.4247618

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
        boruta_trials (int): The number of trials to run. A larger number is
            better as the distribution will be more robust to random fluctuations. 
            Defaults to 50.
        model (str): The ensemble method to use when fitting and calculating
            the feature importance metric. Only two options are currently
            supported, 'rf' for Random Forest and 'xgb' for Extreme Gradient Boosting.
            Defaults to 'rf'.
        importance_type (str): The feature importance type to use, only applicable
            when using clf='xgb'. The options include “gain”, “weight”, “cover”,
            “total_gain” or “total_cover”. Defaults to 'gain'.

    Returns:
        First output is a 1D array containing the indices of the selected features. 
        These indices can then be used to select the columns in the data_x array.
        Second output is the feature selection object, which contains feature selection
        history information and visualization options.
    """
    
    if boruta_trials == 0:
        return np.arange(data_x.shape[1]), None

    if boruta_trials < 20:
        print('WARNING: Results are unstable if boruta_trials is too low!')
    if np.any(np.isnan(data_x)):
        print('NaN values detected, applying Strawman imputation...')
        data_x = Strawman_imputation(data_x)

    if model == 'rf':
        classifier = RandomForestClassifier(random_state=1909)
    elif model == 'xgb':
        classifier = XGBClassifier(tree_method='exact', max_depth=20, importance_type=importance_type, random_state=1909)
    else:
        raise ValueError('Model argument must either be "rf" or "xgb".')
    
    try:
        #BorutaShap program requires input to have the columns attribute
        #Converting to Pandas dataframe
        cols = [str(i) for i in np.arange(data_x.shape[1])]
        X = DataFrame(data_x, columns=cols)
        y = np.zeros(len(data_y))

        #Below is to convert categorical labels to numerical, as per BorutaShap requirements
        for i, label in enumerate(np.unique(data_y)):
            mask = np.where(data_y == label)[0]
            y[mask] = i

        feat_selector = BorutaShap(model=classifier, importance_measure='shap', classification=True)
        print('Running feature selection...')
        feat_selector.fit(X=X, y=y, n_trials=boruta_trials, verbose=False, random_state=1909)

        index = np.array([int(feat) for feat in feat_selector.accepted])
        index.sort()
        print('Feature selection complete, {} selected out of {}!'.format(len(index), data_x.shape[1]))
    except:
        print('Boruta with Shapley values failed, switching to original Boruta...')
        index = boruta_opt(data_x, data_y)

    return index, feat_selector

def boruta_opt(data_x, data_y):
    """
    Applies the Boruta algorithm (Kursa & Rudnicki 2011) to identify features
    that perform worse than random.

    See: https://arxiv.org/pdf/1106.5112.pdf

    Args:
        data_x (ndarray): 2D array of size (n x m), where n is the
            number of samples, and m the number of features.
        data_y (ndarray, str): 1D array containing the corresponing labels.
            
    Returns:
        1D array containing the indices of the selected features. This can then
        be used to index the columns in the data_x array.
    """

    classifier = RandomForestClassifier(random_state=1909)

    feat_selector = BorutaPy(classifier, n_estimators='auto', random_state=1909)
    print('Running feature selection...')
    feat_selector.fit(data_x, data_y)

    feats = np.array([str(feat) for feat in feat_selector.support_])
    index = np.where(feats == 'True')[0]
    print('Feature selection complete, {} selected out of {}!'.format(len(index),len(feat_selector.support)))
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

    if np.all(np.isfinite(data)):
        print('No missing values in data, returning original array.')
        return data 

    if len(data.shape) == 1:
        mask = np.where(np.isfinite(data))[0]
        median = np.median(data[mask])
        data[np.isnan(data)] = median 

        return data

    Ny, Nx = data.shape
    imputed_data = np.zeros((Ny,Nx))

    for i in range(Nx):
        mask = np.where(np.isfinite(data[:,i]))[0]
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

    if imputer is None:
        imputer = KNNImputer(n_neighbors=k)
        imputer.fit(data)
        imputed_data = imputer.transform(data)
        return imputed_data, imputer

    return imputer.transform(data) 

def MissForest_imputation(data):
    """
    !!! THIS ALGORITHM REFITS EVERY TIME, THEREFORE NOT HELPFUL
    FOR IMPUTING NEW, UNSEEN DATA. USE KNN_IMPUTATION INSTEAD !!!

    Imputation algorithm created by Stekhoven and Buhlmann (2012).
    See: https://academic.oup.com/bioinformatics/article/28/1/112/219101

    By default the imputer will be created and returned, unless
    the imputer argument is set, in which case only the transformed
    data is output. 

    Note:
        The RF imputation procedures improve performance if the features are heavily correlated.
        Correlation is important for RF imputation, see: https://arxiv.org/pdf/1701.05305.pdf
    
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

    if np.all(np.isfinite(data)):
        raise ValueError('No missing values in training dataset, do not apply MissForest imputation!')
    
    imputer = MissForest(verbose=0)
    imputer.fit(data)
    imputed_data = imputer.transform(data)

    return imputer.transform(data) 


