#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:40:39 2021

@author: daniel
"""
import os
from pathlib import Path
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import joblib

from optuna.visualization.matplotlib import plot_optimization_history
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy, SquaredHinge
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, MaxPool2D, AveragePooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session 
from keras.models import load_model
from keras.backend import pool2d 

os.environ['CUDA_VISIBLE_DEVICES'], os.environ['TF_CPP_MIN_LOG_LEVEL']= '-1', '3'
from pyBIA.data_processing import process_class, create_training_set, concat_channels
from pyBIA.data_augmentation import augmentation, resize
from pyBIA import optimization


class Classifier:
    """
    Creates and trains the convolutional neural network. 
    The built-in methods can be used predict new samples, and
    also optimize the engine and output visualizations.

    Attributes:
        model: The machine learning model that is created
        
        load: load model

        save: save model

        predict: predict new samples

        plot_hyper_opt: Plot hyperparameter optimization history

    Args:
        blob_data (ndarray, optional): 2D array of size (n x m), where n is the
            number of samples, and m the number of features. Defaults to None,
            in which case a model can be loaded using the class attributes.
        other_data (ndarray, optional): 1D array containing the corresponing labels. Defaults to None,
            in which case a model can be loaded using the class attributes..
        clf (str): The machine learning classifier to optimize. Can either be
            'rf' for Random Forest, 'nn' for Neural Network, or 'xgb' for Extreme Gradient Boosting. 
            Defaults to 'rf'.
        optimize (bool): If True the Boruta algorithm will be run to identify the features
            that contain useful information, after which the optimal Random Forest hyperparameters
            will be calculated using Bayesian optimization. 
        impute (bool): If False no data imputation will be performed. Defaults to True,
            which will result in two outputs, the classifier and the imputer to save
            for future transformations. 
        imp_method (str): The imputation techinque to apply, can either be 'KNN' for k-nearest
            neighbors imputation, or 'MissForest' for the MissForest machine learning imputation
            algorithm. Defaults to 'KNN'.
        n_iter (int): The maximum number of iterations to perform during 
            the hyperparameter search. Defaults to 25. 

    Returns:
        Trained machine learning model.

    """
    def __init__(self, blob_data=None, other_data=None, val_blob=None, val_other=None, pyBIA_model=1, img_num_channels=1, 
        optimize=True, n_iter=25, normalize=True, min_pixel=638, max_pixel=3000, epochs=100, train_epochs=25, 
        patience=5, opt_model=True, opt_aug=False, batch_min=10, batch_max=250, image_size_min=50, image_size_max=90, balance_val=True,
        opt_max_min_pix=None, opt_max_max_pix=None, metric='loss'):

        self.blob_data = blob_data
        self.other_data = other_data
        self.optimize = optimize 
        self.metric = metric 
        self.n_iter = n_iter
        self.pyBIA_model = pyBIA_model
        self.img_num_channels = img_num_channels
        self.normalize = normalize 
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.val_blob = val_blob
        self.val_other = val_other
        self.epochs = epochs
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

        self.model = None
        self.history = None 
        self.best_params = None 
        self.optimization_results = None 

    def create(self):
        """
        Creates the CNN machine learning engine.
        
        Returns:
            Trained classifier.
        """

        if self.optimize is False:
            print("Returning base model number {}...".format(self.pyBIA_model))
            if self.pyBIA_model == 1:
                self.model, self.history = pyBIA_model_1(self.blob_data, self.other_data, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_blob=self.val_blob, val_other=self.val_other, epochs=self.epochs)
            elif self.pyBIA_model == 2:
                self.model, self.history = pyBIA_model_2(self.blob_data, self.other_data, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_blob=self.val_blob, val_other=self.val_other, epochs=self.epochs)
            elif self.pyBIA_model == 3:
                self.model, self.history = pyBIA_model_3(self.blob_data, self.other_data, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_blob=self.val_blob, val_other=self.val_other, epochs=self.epochs)
            elif self.pyBIA_model == 4:
                self.model, self.history = pyBIA_model_4(self.blob_data, self.other_data, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_blob=self.val_blob, val_other=self.val_other, epochs=self.epochs)
            elif self.pyBIA_model == 5:
                self.model, self.history = pyBIA_model_5(self.blob_data, self.other_data, img_num_channels=self.img_num_channels, normalize=self.normalize,
                    min_pixel=self.min_pixel, max_pixel=self.max_pixel, val_blob=self.val_blob, val_other=self.val_other, epochs=self.epochs)
            else:
                raise ValueError('Invalid input -- pyBIA_model parameter must be an integer between 1 and 5.')
            
            return      

        self.best_params, self.optimization_results = optimization.hyper_opt(self.blob_data, self.other_data, clf='cnn', metric=self.metric, n_iter=self.n_iter, 
            balance=False, return_study=True, img_num_channels=self.img_num_channels, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, 
            val_X=self.val_blob, val_Y=self.val_other, train_epochs=self.train_epochs, patience=self.patience, opt_model=self.opt_model, opt_aug=self.opt_aug, batch_min=self.batch_min, 
            batch_max=self.batch_max, image_size_min=self.image_size_min, image_size_max=self.image_size_max, balance_val=self.balance_val,
            opt_max_min_pix=self.opt_max_min_pix, opt_max_max_pix=self.opt_max_max_pix, pyBIA_model=self.pyBIA_model)

        if self.epochs != 0:
            print("Fitting and returning final model...")
            clear_session()

            if self.opt_max_min_pix is not None:
                min_pix, max_pix = 0.0, self.best_params['max_pixel']
                self.normalize=True
            else:
                min_pix, max_pix = self.min_pixel, self.max_pixel

            if self.opt_aug:
                if self.img_num_channels == 1:
                    channel1, channel2, channel3 = self.blob_data, None, None 
                elif self.img_num_channels == 2:
                    channel1, channel2, channel3 = self.blob_data[:,:,:,0], self.blob_data[:,:,:,1], None 
                elif self.img_num_channels == 3:
                    channel1, channel2, channel3 = self.blob_data[:,:,:,0], self.blob_data[:,:,:,1], self.blob_data[:,:,:,2]
                else:
                    raise ValueError('Only three filters are supported!')

                augmented_images = augmentation(channel1=channel1, channel2=channel2, channel3=channel3, batch=self.best_params['batch'], 
                    width_shift=self.best_params['shift'], height_shift=self.best_params['shift'], horizontal=self.best_params['horizontal'], 
                    vertical=self.best_params['vertical'], rotation=self.best_params['rotation'], image_size=self.best_params['image_size'])

                if self.img_num_channels > 1:
                    class_1=[]
                    if self.img_num_channels == 2:
                        for i in range(len(augmented_images[0])):
                            class_1.append(concat_channels(augmented_images[0][i], augmented_images[1][i]))
                    else:
                        for i in range(len(augmented_images[0])):
                            class_1.append(concat_channels(augmented_images[0][i], augmented_images[1][i], augmented_images[2][i]))
                    class_1 = np.array(class_1)
                else:
                    class_1 = augmented_images

                if self.balance_val:
                    class_2 = self.other_data[:len(class_1)]   
                else:
                    class_2 = self.other_data   

                if self.img_num_channels == 1:
                    class_2 = resize(class_2, size=image_size)
                else:
                    channel1 = resize(class_2[:,:,:,0], size=image_size)
                    channel2 = resize(class_2[:,:,:,1], size=image_size)
                    if self.img_num_channels == 2:
                        class_2 = concat_channels(channel1, channel2)
                    else:
                        channel3 = resize(class_2[:,:,:,2], size=image_size)
                        class_2 = concat_channels(channel1, channel2, channel3)

                #Need to also crop the validation images
                if self.val_blob is not None:
                    if self.img_num_channels == 1:
                        val_class_1 = resize(self.val_blob, size=image_size)
                    else:
                        val_channel1 = resize(self.val_blob[:,:,:,0], size=image_size)
                        val_channel2 = resize(self.val_blob[:,:,:,1], size=image_size)
                        if self.img_num_channels == 2:
                            val_class_1 = concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_blob[:,:,:,2], size=image_size)
                            val_class_1 = concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_1 = None 

                if self.val_other is not None:
                    if self.img_num_channels == 1:
                        val_class_2 = resize(self.val_other, size=image_size)
                    elif self.img_num_channels > 1:
                        val_channel1 = resize(self.val_other[:,:,:,0], size=image_size)
                        val_channel2 = resize(self.val_other[:,:,:,1], size=image_size)
                        if self.img_num_channels == 2:
                            val_class_2 = concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_other[:,:,:,2], size=image_size)
                            val_class_2 = concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_2 = None 

            else:
                class_1, class_2 = self.blob_data, self.other_data
                val_class_1, val_class_2 = self.val_blob, self.val_other

            if self.opt_model:
                if self.pyBIA_model == 1:
                    model, history = cnn_model.pyBIA_model_1(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.train_epochs, batch_size=self.best_params['batch_size'], 
                        lr=self.best_params['lr'], momentum=self.best_params['momentum'], decay=self.best_params['decay'], nesterov=self.best_params['nesterov'], 
                        activation_conv=self.best_params['activation_conv'], activation_dense=self.best_params['activation_dense'], pooling_1=self.best_params['pooling_1'], 
                        pool_size_1=self.best_params['pool_size_1'], pool_stride_1=self.best_params['pool_stride_1'], filter_1=self.best_params['filter_1'], 
                        filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], dense_neurons_1=self.best_params['dense_neurons_1'], 
                        dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'])

                elif self.pyBIA_model == 2:
                    model, history = cnn_model.pyBIA_model_2(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.train_epochs, batch_size=self.best_params['batch_size'], 
                        lr=self.best_params['lr'], momentum=self.best_params['momentum'], decay=self.best_params['decay'], nesterov=self.best_params['nesterov'], 
                        activation_conv=self.best_params['activation_conv'], activation_dense=self.best_params['activation_dense'], pooling_1=self.best_params['pooling_1'], 
                        pooling_2=self.best_params['pooling_2'], pool_size_1=self.best_params['pool_size_1'], pool_stride_1=self.best_params['pool_stride_1'], 
                        pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'], filter_1=self.best_params['filter_1'], 
                        filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], filter_2=self.best_params['filter_2'], 
                        filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], dense_neurons_1=self.best_params['dense_neurons_1'], 
                        dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'])

                elif self.pyBIA_model == 3:
                    model, history = cnn_model.pyBIA_model_3(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.train_epochs, batch_size=self.best_params['batch_size'], 
                        lr=self.best_params['lr'], momentum=self.best_params['momentum'], decay=self.best_params['decay'], nesterov=self.best_params['nesterov'], 
                        activation_conv=self.best_params['activation_conv'], activation_dense=self.best_params['activation_dense'], pooling_1=self.best_params['pooling_1'], 
                        pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'], pool_size_1=self.best_params['pool_size_1'], 
                        pool_stride_1=self.best_params['pool_stride_1'], pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'], 
                        pool_size_3=self.best_params['pool_size_3'], pool_stride_3=self.best_params['pool_stride_3'], filter_1=self.best_params['filter_1'], 
                        filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], filter_2=self.best_params['filter_2'], 
                        filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], filter_3=self.best_params['filter_3'], 
                        filter_size_3=self.best_params['filter_size_3'], strides_3=self.best_params['strides_3'], dense_neurons_1=self.best_params['dense_neurons_1'], 
                        dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'])
                
                elif self.pyBIA_model == 4:
                    model, history = cnn_model.pyBIA_model_4(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.train_epochs, batch_size=self.best_params['batch_size'], 
                        lr=self.best_params['lr'], momentum=self.best_params['momentum'], decay=self.best_params['decay'], nesterov=self.best_params['nesterov'], 
                        activation_conv=self.best_params['activation_conv'], activation_dense=self.best_params['activation_dense'], pooling_1=self.best_params['pooling_1'], 
                        pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'], pool_size_1=self.best_params['pool_size_1'], 
                        pool_stride_1=self.best_params['pool_stride_1'], pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'], 
                        pool_size_3=self.best_params['pool_size_3'], pool_stride_3=self.best_params['pool_stride_3'], filter_1=self.best_params['filter_1'], 
                        filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], filter_2=self.best_params['filter_2'], 
                        filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], filter_3=self.best_params['filter_3'], 
                        filter_size_3=self.best_params['filter_size_3'], strides_3=self.best_params['strides_3'], filter_4=self.best_params['filter_4'], 
                        filter_size_4=self.best_params['filter_size_4'], strides_4=self.best_params['strides_4'], dense_neurons_1=self.best_params['dense_neurons_1'], 
                        dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'])
                
                elif self.pyBIA_model == 5:
                    self.model, self.history = cnn_model.pyBIA_model_5(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize, 
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.train_epochs, batch_size=self.best_params['batch_size'], 
                        lr=self.best_params['lr'], momentum=self.best_params['momentum'], decay=self.best_params['decay'], nesterov=self.best_params['nesterov'], 
                        activation_conv=self.best_params['activation_conv'], activation_dense=self.best_params['activation_dense'], pooling_1=self.best_params['pooling_1'], 
                        pooling_2=self.best_params['pooling_2'], pooling_3=self.best_params['pooling_3'], pool_size_1=self.best_params['pool_size_1'], 
                        pool_stride_1=self.best_params['pool_stride_1'], pool_size_2=self.best_params['pool_size_2'], pool_stride_2=self.best_params['pool_stride_2'], 
                        pool_size_3=self.best_params['pool_size_3'], pool_stride_3=self.best_params['pool_stride_3'], filter_1=self.best_params['filter_1'], 
                        filter_size_1=self.best_params['filter_size_1'], strides_1=self.best_params['strides_1'], filter_2=self.best_params['filter_2'], 
                        filter_size_2=self.best_params['filter_size_2'], strides_2=self.best_params['strides_2'], filter_3=self.best_params['filter_3'], 
                        filter_size_3=self.best_params['filter_size_3'], strides_3=self.best_params['strides_3'], filter_4=self.best_params['filter_4'], 
                        filter_size_4=self.best_params['filter_size_4'], strides_4=self.best_params['strides_4'], filter_5=self.best_params['filter_5'], 
                        filter_size_5=self.best_params['filter_size_5'], strides_5=self.best_params['strides_5'], dense_neurons_1=self.best_params['dense_neurons_1'], 
                        dense_neurons_2=self.best_params['dense_neurons_2'], dropout_1=self.best_params['dropout_1'], dropout_2=self.best_params['dropout_2'])
            
            else: 
                if self.pyBIA_model == 1:
                    self.model, self.history = pyBIA_model_1(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.epochs)
                elif self.pyBIA_model == 2:
                    self.model, self.history = pyBIA_model_2(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.epochs)
                elif self.pyBIA_model == 3:
                    self.model, self.history = pyBIA_model_3(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.epochs)
                elif self.pyBIA_model == 4:
                    self.model, self.history = pyBIA_model_4(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.epochs)
                elif self.pyBIA_model == 5:
                    self.model, self.history = pyBIA_model_5(class_1, class_2, img_num_channels=self.img_num_channels, normalize=self.normalize,
                        min_pixel=min_pix, max_pixel=max_pix, val_blob=val_class_1, val_other=val_class_2, epochs=self.epochs)
           
        return 

    def save(self, path=None, overwrite=False):
        """
        Saves the trained classifier in a new directory named 'pyBIA_models', 
        as well as the imputer and the features to use, if applicable.
        
        Args:
            path (str): Absolute path where the data folder will be saved
                Defaults to None, in which case the directory is saved to the
                local home directory.
            overwrite (bool, optional): If True the 'pyBIA_models' folder this
                function creates in the specified path will be deleted if it exists
                and created anew to avoid duplicate files. 
        """

        if self.model is None:
            print('The model has not been created! Run classifier.create() first.')

        if path is None:
            path = str(Path.home())
        if path[-1] != '/':
            path+='/'

        try:
            os.mkdir(path+'pyBIA_cnn_model')
        except FileExistsError:
            if overwrite:
                try:
                    os.rmdir(path+'pyBIA_cnn_model')
                except OSError:
                    for file in os.listdir(path+'pyBIA_cnn_model'):
                        os.remove(path+'pyBIA_cnn_model/'+file)
                    os.rmdir(path+'pyBIA_cnn_model')
                os.mkdir(path+'pyBIA_cnn_model')
            else:
                raise ValueError('Tried to create "pyBIA_cnn_model" directory in specified path but folder already exists! If you wish to overwrite set overwrite=True.')
        
        path += 'pyBIA_cnn_model/'
        if self.model is not None:      
            np.savetxt(path+'model_acc', self.history.history['accuracy'])
            np.savetxt(path+'model_loss', self.history.history['loss'])
            if self.val_blob is not None:
                np.savetxt(path+'model_val_acc', self.history.history['val_accuracy'])
                np.savetxt(path+'model_val_loss', self.history.history['val_loss'])

            save_model(self.model, path+'Keras_Model.h5')

        if self.best_params is not None:
            joblib.dump(self.best_params, path+'Best_Params')
        if self.optimization_results is not None:
            joblib.dump(self.optimization_results, path+'HyperOpt_Results')
        print('Files saved in: {}'.format(path))

        return 

    def load(self, path=None):
        """ 
        Loads the model, imputer, and feats to use, if created and saved.
        This function will look for a folder named 'pyBIA_models' in the
        local home directory, unless a path argument is set. 

        Args:
            path (str): Path where the directory 'pyBIA_models' is saved. 
            Defaults to None, in which case the folder is assumed to be in the 
            local home directory.
        """

        if path is None:
            path = str(Path.home())
        if path[-1] != '/':
            path+='/'

        path += 'pyBIA_cnn_model/'

        try:
            self.model = load_model(path+'Keras_Model.h5')
            model = 'model'
        except:
            model = ''
            pass

        try:
            self.optimization_results = joblib.load(path+'HyperOpt_Results')
            optimization_results = 'optimization_results'
        except:
            optimization_results = '' 
            pass

        try:
            self.best_params = joblib.load(path+'Best_Params')
            best_params = 'best_params'
        except:
            best_params = '' 
            pass        

        print('Successfully loaded the following class attributes: {}, {}, {}'.format(model, optimization_results, best_params))
        
        return

    def predict(self, data, target='DIFFUSE', return_proba=False):
        """
        Returns the class prediction. The input can either be a single 2D array 
        or a 3D array if there are multiple samples.

        Args:
            data: 2D array for single image, 3D array for multiple images.
            target (str): The name of the target class, assuming binary classification in 
                which there is an 'OTHER' class. Defaults to 'DIFFUSE'. 
            return_proba (bool): If True the output will return the probability prediction.
                Defaults to False. 
        Returns:
            The class prediction(s), either 'DIFFUSE' or 'OTHER'.
        """
      
        data = process_class(data, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)
        predictions = self.model.predict(data)

        output=[]
        probas=[]
        for i in range(len(predictions)):
            if np.argmax(predictions[i]) == 0:
                prediction = target
                probas.append(predictions[i][0])
            else:
                prediction = 'OTHER'
                probas.append(predictions[i][1])

            output.append(prediction)

        if return_proba:
            output = np.c_[output, probas]
            
        return np.array(output)

    def plot_hyper_opt(self, baseline=None, xlim=None, ylim=None, xlog=True, ylog=False, savefig=False):
        """
        Plots the hyperparameter optimization history.
    
        Args:
            baseline (float): Baseline accuracy achieved when using only
                the default engine hyperparameters. If input a vertical
                line will be plot to indicate this baseline accuracy.
                Defaults to None.
            xlim: Limits for the x-axis. Ex) xlim = (0, 1000)
            ylim: Limits for the y-axis. Ex) ylim = (0.9, 0.94)
            xlog (boolean): If True the x-axis will be log-scaled.
                Defaults to True.
            ylog (boolean): If True the y-axis will be log-scaled.
                Defaults to False.


        Returns:
            AxesImage
        """

        fig = plot_optimization_history(self.optimization_results)
        if baseline is not None:
            plt.axhline(y=baseline, color='k', linestyle='--', linewidth=3, alpha=0.7, label='Default Model')
        
        if self.metric == 'val_accuracy':
            ylabel = 'Validation Accuracy'
        elif self.metric == 'accuracy':
            ylabel = 'Training Accuracy'
        elif self.metric == 'val_loss':
            ylabel = '1 - Validation Loss'
        elif self.metric == 'loss':
            ylabel = '1 - Training Loss'
        else:
            ylabel = 'Optimization Metric'

        plt.xlabel('Trial #', size=16 alpha=1, color='k')
        plt.ylabel(ylabel, size=16 alpha=1, color='k')
        plt.title(('Hyperparameter Optimization History'), size=18)
        plt.xticks(fontsize=14, color='k')
        plt.yticks(fontsize=14, color='k')
        plt.grid(True, color='k', alpha=0.35, linewidth=1.5, linestyle='--')
        plt.legend(prop={'size': 12})
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        plt.tight_layout()
        if savefig:
            plt.savefig('CNN_Hyperparameter_Optimization.png', bbox_inches='tight', dpi=300)
        else:
            plt.show()

    #def load_bw_model(self):
    #    """
    #    Calling this will load the trained Tensorflow model, trained using NDWFS images
    #    in the blue broadband.
    #    
    #    Note:
    #        Training new models with 1000 epochs can take over a week, this Bw model
    #        was trained using NDWFS blue broadband images of the Bootes field. The 
    #        corresponding .h5 file is located in the data folder inside the pyBIA directory 
    #       in the Python path. 
    #
    #    Returns:
    #        The pyBIA CNN model used for classifying images in blue broadband surveys.

    #  """

    #   resource_package = __name__
    #   resource_path = '/'.join(('data', 'Bw_CNN_Model.h5'))
    #   self.model = load_model(pkg_resources.resource_filename(resource_package, resource_path))
    #   print('Bw model successfully loaded.')
    #   print('Note: Input data when using this model must be 50x50.')
    #   return 

def pyBIA_model_1(blob_data, other_data, img_num_channels=1, normalize=True, 
        min_pixel=0, max_pixel=100, val_blob=None, val_other=None, epochs=100, 
        batch_size=32, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, 
        loss='categorical_crossentropy', activation_conv='relu', activation_dense='tanh', 
        padding='same', pooling_1='max', pool_size_1=3, pool_stride_1=2, filter_1=96, 
        filter_size_1=11, strides_1=4, dense_neurons_1=4096, dense_neurons_2=4096,
        dropout_1=0.5, dropout_2=0.5, early_stop_callback=None, checkpoint=True):
        """
        The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
        Visual Recognition Challenge, AlexNet. Parameters were adapted for
        our astronomy case of detecting diffuse emission.

        Args:
            pooling (str): Defines the pooling method to perform, can either be
                'max', 'min', 'average', or None

        Returns:
            The trained CNN model.
        """

        if len(blob_data.shape) != len(other_data.shape):
            raise ValueError("Shape of blob and other data must be the same.")
        if batch_size < 16:
            warn("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.", stacklevel=2)
        if val_blob is not None:
            val_X1, val_Y1 = process_class(val_blob, label=0, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
            if val_other is None:
                val_X, val_Y = val_X1, val_Y1
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        else:
            if val_other is None:
                val_X, val_Y = None, None
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = val_X2, val_Y2

        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]
        #decay = lr / epochs

        ix = np.random.permutation(len(blob_data))
        blob_data = blob_data[ix]

        ix = np.random.permutation(len(other_data))
        other_data = other_data[ix]

        X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

        if normalize:
            X_train[X_train > 1] = 1
            X_train[X_train < 0] = 0
            
        input_shape = (img_width, img_height, img_num_channels)
       
        # Uniform scaling initializer
        num_classes = 2
        uniform_scaling = VarianceScaling(
            scale=1.0, mode='fan_in', distribution='uniform', seed=None)

        if loss == 'squared_hinge':
            loss = SquaredHinge()

        # Model configuration
        model = Sequential()

        #Convolutional layers
        model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape,
                         padding=padding, kernel_initializer=uniform_scaling))
        if pooling_1 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))
        elif pooling_1 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))            
        model.add(BatchNormalization())

        #FCC
        model.add(Flatten())
        model.add(Dense(dense_neurons_1, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_1))
        model.add(Dense(dense_neurons_2, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_2))

        #Output layer
        model.add(Dense(num_classes, activation='softmax',
                        kernel_initializer='TruncatedNormal'))

        optimizer = SGD(learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        path = str(Path.home())+'/'
        callbacks_list = []
        if checkpoint:
            model_checkpoint = ModelCheckpoint(path+'checkpoint.hdf5', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
            callbacks_list.append(model_checkpoint)
        if early_stop_callback is not None:
            callbacks_list.append(early_stop_callback)

        if val_X is None:
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1)
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=1)

        return model, history


def pyBIA_model_2(blob_data, other_data, img_num_channels=1, normalize=True, 
        min_pixel=0, max_pixel=100, val_blob=None, val_other=None, epochs=100, 
        batch_size=32, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, 
        loss='categorical_crossentropy', activation_conv='relu', activation_dense='tanh', 
        padding='same', pooling_1='max', pooling_2='max', pool_size_1=3, pool_stride_1=2, 
        pool_size_2=3, pool_stride_2=2, filter_1=96, filter_size_1=11, strides_1=4, 
        filter_2=256, filter_size_2=5, strides_2=1, dense_neurons_1=4096, dense_neurons_2=4096,
        dropout_1=0.5, dropout_2=0.5, early_stop_callback=None, checkpoint=True):
        """
        The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
        Visual Recognition Challenge, AlexNet. Parameters were adapted for
        our astronomy case of detecting diffuse emission.

        Returns:
            The trained CNN model.
        """
        
        if len(blob_data.shape) != len(other_data.shape):
            raise ValueError("Shape of blob and other data must be the same.")
        if batch_size < 16:
            warn("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.", stacklevel=2)
        if val_blob is not None:
            val_X1, val_Y1 = process_class(val_blob, label=0, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
            if val_other is None:
                val_X, val_Y = val_X1, val_Y1
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        else:
            if val_other is None:
                val_X, val_Y = None, None
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = val_X2, val_Y2

        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]
        #decay = lr / epochs

        ix = np.random.permutation(len(blob_data))
        blob_data = blob_data[ix]

        ix = np.random.permutation(len(other_data))
        other_data = other_data[ix]

        X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

        if normalize:
            X_train[X_train > 1] = 1
            X_train[X_train < 0] = 0
            
        input_shape = (img_width, img_height, img_num_channels)
       
        # Uniform scaling initializer
        num_classes = 2
        uniform_scaling = VarianceScaling(
            scale=1.0, mode='fan_in', distribution='uniform', seed=None)

        if loss == 'squared_hinge':
            loss = SquaredHinge()

        # Model configuration
        model = Sequential()

        #Convolutional layers
        model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape,
                         padding=padding, kernel_initializer=uniform_scaling))
        if pooling_1 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))
        elif pooling_1 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))            
        model.add(BatchNormalization())

        model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        if pooling_2 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))
        elif pooling_2 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))            
        model.add(BatchNormalization())

        #FCC
        model.add(Flatten())
        model.add(Dense(dense_neurons_1, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_1))
        model.add(Dense(dense_neurons_2, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_2))

        #Output layer
        model.add(Dense(num_classes, activation='softmax',
                        kernel_initializer='TruncatedNormal'))

        optimizer = SGD(learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        path = str(Path.home())+'/'
        callbacks_list = []
        if checkpoint:
            model_checkpoint = ModelCheckpoint(path+'checkpoint.hdf5', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
            callbacks_list.append(model_checkpoint)
        if early_stop_callback is not None:
            callbacks_list.append(early_stop_callback)

        if val_X is None:
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1)
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=1)

        return model, history

def pyBIA_model_3(blob_data, other_data, img_num_channels=1, normalize=True, 
        min_pixel=0, max_pixel=100, val_blob=None, val_other=None, epochs=100, 
        batch_size=32, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, 
        loss='categorical_crossentropy', activation_conv='relu', activation_dense='tanh', 
        padding='same', pooling_1='max', pooling_2='max', pooling_3='max', 
        pool_size_1=3, pool_stride_1=2, pool_size_2=3, pool_stride_2=2, pool_size_3=3, pool_stride_3=2, 
        filter_1=96, filter_size_1=11, strides_1=4, filter_2=256, filter_size_2=5, strides_2=1,
        filter_3=384, filter_size_3=3, strides_3=1, dense_neurons_1=4096, dense_neurons_2=4096,
        dropout_1=0.5, dropout_2=0.5, early_stop_callback=None, checkpoint=True):
        """
        The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
        Visual Recognition Challenge, AlexNet. Parameters were adapted for
        our astronomy case of detecting diffuse emission.
            
        Returns:
            The trained CNN model.
        """
        
        if len(blob_data.shape) != len(other_data.shape):
            raise ValueError("Shape of blob and other data must be the same.")
        if batch_size < 16:
            warn("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.", stacklevel=2)
        if val_blob is not None:
            val_X1, val_Y1 = process_class(val_blob, label=0, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
            if val_other is None:
                val_X, val_Y = val_X1, val_Y1
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        else:
            if val_other is None:
                val_X, val_Y = None, None
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = val_X2, val_Y2

        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]
        #decay = lr / epochs

        ix = np.random.permutation(len(blob_data))
        blob_data = blob_data[ix]

        ix = np.random.permutation(len(other_data))
        other_data = other_data[ix]

        X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

        if normalize:
            X_train[X_train > 1] = 1
            X_train[X_train < 0] = 0
            
        input_shape = (img_width, img_height, img_num_channels)
       
        # Uniform scaling initializer
        num_classes = 2
        uniform_scaling = VarianceScaling(
            scale=1.0, mode='fan_in', distribution='uniform', seed=None)

        if loss == 'squared_hinge':
            loss = SquaredHinge()

        # Model configuration
        model = Sequential()

        #Convolutional layers
        model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape,
                         padding=padding, kernel_initializer=uniform_scaling))
        if pooling_1 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))
        elif pooling_1 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))            
        model.add(BatchNormalization())

        model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        if pooling_2 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))
        elif pooling_2 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))            
        model.add(BatchNormalization())

        model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        if pooling_3 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))
        elif pooling_3 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))            
        model.add(BatchNormalization())

        #FCC
        model.add(Flatten())
        model.add(Dense(dense_neurons_1, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_1))
        model.add(Dense(dense_neurons_2, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_2))

        #Output layer
        model.add(Dense(num_classes, activation='softmax',
                        kernel_initializer='TruncatedNormal'))

        optimizer = SGD(learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        path = str(Path.home())+'/'
        callbacks_list = []
        if checkpoint:
            model_checkpoint = ModelCheckpoint(path+'checkpoint.hdf5', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
            callbacks_list.append(model_checkpoint)
        if early_stop_callback is not None:
            callbacks_list.append(early_stop_callback)

        if val_X is None:
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1)
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=1)

        return model, history

def pyBIA_model_4(blob_data, other_data, img_num_channels=1, normalize=True, 
        min_pixel=0, max_pixel=100, val_blob=None, val_other=None, epochs=100, 
        batch_size=32, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, 
        loss='categorical_crossentropy', activation_conv='relu', activation_dense='tanh', 
        padding='same', pooling_1='max', pooling_2='max', pooling_3='max', pool_size_1=3, 
        pool_stride_1=2, pool_size_2=3, pool_stride_2=2, pool_size_3=3, pool_stride_3=2, 
        filter_1=96, filter_size_1=11, strides_1=4, filter_2=256, filter_size_2=5, strides_2=1,
        filter_3=384, filter_size_3=3, strides_3=1, filter_4=384, filter_size_4=3, strides_4=1,
        dense_neurons_1=4096, dense_neurons_2=4096, dropout_1=0.5, dropout_2=0.5, 
        early_stop_callback=None, checkpoint=True):
        """
        The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
        Visual Recognition Challenge, AlexNet. Parameters were adapted for
        our astronomy case of detecting diffuse emission.
            
        Returns:
            The trained CNN model.
        """
        
        if len(blob_data.shape) != len(other_data.shape):
            raise ValueError("Shape of blob and other data must be the same.")
        if batch_size < 16:
            warn("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.", stacklevel=2)
        if val_blob is not None:
            val_X1, val_Y1 = process_class(val_blob, label=0, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
            if val_other is None:
                val_X, val_Y = val_X1, val_Y1
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        else:
            if val_other is None:
                val_X, val_Y = None, None
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = val_X2, val_Y2

        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]
        #decay = lr / epochs

        ix = np.random.permutation(len(blob_data))
        blob_data = blob_data[ix]

        ix = np.random.permutation(len(other_data))
        other_data = other_data[ix]

        X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

        if normalize:
            X_train[X_train > 1] = 1
            X_train[X_train < 0] = 0
            
        input_shape = (img_width, img_height, img_num_channels)
       
        # Uniform scaling initializer
        num_classes = 2
        uniform_scaling = VarianceScaling(
            scale=1.0, mode='fan_in', distribution='uniform', seed=None)

        if loss == 'squared_hinge':
            loss = SquaredHinge()

        # Model configuration
        model = Sequential()

        #Convolutional layers
        model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape,
                         padding=padding, kernel_initializer=uniform_scaling))
        if pooling_1 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))
        elif pooling_1 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))            
        model.add(BatchNormalization())

        model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        if pooling_2 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))
        elif pooling_2 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))            
        model.add(BatchNormalization())

        model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        if pooling_3 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))
        elif pooling_3 == 'min':
            model.add(MinPool2D(pool_size=(pool_size_3,pool_size_3), strides=(pool_stride_3,pool_stride_3), padding=padding))
        elif pooling_3 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))            
        model.add(BatchNormalization())

        #FCC
        model.add(Flatten())
        model.add(Dense(dense_neurons_1, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_1))
        model.add(Dense(dense_neurons_2, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_2))

        #Output layer
        model.add(Dense(num_classes, activation='softmax',
                        kernel_initializer='TruncatedNormal'))

        optimizer = SGD(learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        path = str(Path.home())+'/'
        callbacks_list = []
        if checkpoint:
            model_checkpoint = ModelCheckpoint(path+'checkpoint.hdf5', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
            callbacks_list.append(model_checkpoint)
        if early_stop_callback is not None:
            callbacks_list.append(early_stop_callback)

        if val_X is None:
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1)
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=1)

        return model, history


def pyBIA_model_5(blob_data, other_data, img_num_channels=1, normalize=True, 
        min_pixel=0, max_pixel=100, val_blob=None, val_other=None, epochs=100, 
        batch_size=32, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, 
        loss='categorical_crossentropy', activation_conv='relu', activation_dense='tanh', 
        padding='same', pooling_1='max', pooling_2='max', pooling_3='max', 
        pool_size_1=3, pool_stride_1=2, pool_size_2=3, pool_stride_2=2, pool_size_3=3, pool_stride_3=2, 
        filter_1=96, filter_size_1=11, strides_1=4, filter_2=256, filter_size_2=5, strides_2=1,
        filter_3=384, filter_size_3=3, strides_3=1, filter_4=384, filter_size_4=3, strides_4=1,
        filter_5=256, filter_size_5=3, strides_5=1, dense_neurons_1=4096, dense_neurons_2=4096, 
        dropout_1=0.5, dropout_2=0.5, early_stop_callback=None, checkpoint=True):
        """
        The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
        Visual Recognition Challenge, AlexNet. Parameters were adapted for
        our astronomy case of detecting diffuse emission.

        Note:
            To avoid exploding gradients we need to normalize our pixels to be 
            between 0 and 1. By default normalize=True, which will perform
            min-max normalization using the min_pixel and max_pixel arguments, 
            which should be set carefully.

            The min_pixel parameter is set to 0 by default as the data is assumed
            to be background-subtracted. The max_pixel must be adequately brighter
            than the brighest expected target object. In this example we expected
            the high redshift Lyman-alpha nebulae to appear diffuse and less bright,
            so anything brighter than max_pixel=3000 can be categorized as too bright 
            to be a candidate source.
        
        Args:
            blob_data (ndarray): 3D array containing more than one image of diffuse objects.
            other_data (ndarray): 3D array containing more than one image of non-diffuse objects.
            img_num_channels (int): The number of filters used. Defaults to 1, as pyBIA version 1
                has been trained with only blue broadband data.
            normalize (bool, optional): If True the data will be min-max normalized using the 
                input min and max pixels. Defaults to True.
            min_pixel (int, optional): The minimum pixel count, defaults to 638. 
                Pixels with counts below this threshold will be set to this limit.
            max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
                Pixels with counts above this threshold will be set to this limit.
            val_blob (array, optional): 3D matrix containing the 2D arrays (images)
                to be used for validationm, for the blob class. Defaults to None.
            val_other (array, optional): 3D matrix containing the 2D arrays (images)
                to be used for validationm, for the blob class. Defaults to None.
            epochs (int): Number of epochs used for training. 
            batch_size (int): The size of each sub-sample used during the training
                epoch. Large batches are likely to get stuck in local minima. Defaults to 32.
            lr (float): Learning rate, the rate at which the model updates the gradient. Defaults to 0.0001
            momentum (float): Momentum is a float greater than 0 that accelerates gradient descent. Defaults to 0.9.
            decay (float): The rate of learning rate decay applied after each epoch. Defaults to 0.0005. It is recommended
                to set decay to the learning rate divded by the total number of epochs.
            nesterov (bool): Whether to apply Nesterov momentum or not. Defaults to False.
            loss (str): The loss function used to calculate the gradients. Defaults to 'categorical_crossentropy'.
                Loss functions can be set by calling the Keras API losses module.
            activation_conv (str): Activation function to use for the convolutional layer. Default is 'relu'.'
            activation_dense (str): Activation function to use for the dense layers. Default is 'tanh'.
            padding (str): Either 'same' or 'valid'. When set to 'valid', the dimensions reduce as the boundary 
                that doesn't make it within even convolutions get cuts off. Defaults to 'same', which applies
                zero-value padding around the boundary, ensuring even convolutional steps across each dimension.
            dropout (float): Droupout rate after the dense layers. This is the percentage of dense neurons
                that are turned off at each epoch. This prevents inter-neuron depedency, and thus overfitting. 
            pooling (bool): True to enable max pooling, false to disable. 
                Note: Max pooling can result in loss of positional information, it computation allows
                setting pooling=False may yield more robust accuracy.
            pool_size (int, optional): The pool size of the max pooling layers. Defaults to 3.
            pool_stride (int, optional): The stride to use in the max pooling layers. Defaults to 2.
            early_stop_callback (list, optional): Callbacks for early stopping and pruning with Optuna, defaults
                to None. Should only be used during optimization, refer to pyBIA.optimization.objective_cnn().
            checkpoint (bool, optional): If False no checkpoint will be saved. Defaults to True.
            
        Returns:
            The trained CNN model.
        """
        
        if len(blob_data.shape) != len(other_data.shape):
            raise ValueError("Shape of blob and other data must be the same.")
        if batch_size < 16:
            warn("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.", stacklevel=2)
        if val_blob is not None:
            val_X1, val_Y1 = process_class(val_blob, label=0, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
            if val_other is None:
                val_X, val_Y = val_X1, val_Y1
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        else:
            if val_other is None:
                val_X, val_Y = None, None
            else:
                val_X2, val_Y2 = process_class(val_other, label=1, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
                val_X, val_Y = val_X2, val_Y2

        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]
        #decay = lr / epochs

        ix = np.random.permutation(len(blob_data))
        blob_data = blob_data[ix]

        ix = np.random.permutation(len(other_data))
        other_data = other_data[ix]

        X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

        if normalize:
            X_train[X_train > 1] = 1
            X_train[X_train < 0] = 0
            
        input_shape = (img_width, img_height, img_num_channels)
       
        # Uniform scaling initializer
        num_classes = 2
        uniform_scaling = VarianceScaling(
            scale=1.0, mode='fan_in', distribution='uniform', seed=None)

        if loss == 'squared_hinge':
            loss = SquaredHinge()

        # Model configuration
        model = Sequential()

        #Convolutional layers
        model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=activation_conv, input_shape=input_shape,
                         padding=padding, kernel_initializer=uniform_scaling))
        if pooling_1 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))
        elif pooling_1 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding))            
        model.add(BatchNormalization())

        model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        if pooling_2 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))
        elif pooling_2 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding))            
        model.add(BatchNormalization())

        model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        model.add(BatchNormalization())
        model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=activation_conv, padding=padding,
                         kernel_initializer=uniform_scaling))
        if pooling_3 == 'max':
            model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))
        elif pooling_3 == 'average':
            model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding))            
        model.add(BatchNormalization())

        #FCC
        model.add(Flatten())
        model.add(Dense(dense_neurons_1, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_1))
        model.add(Dense(dense_neurons_2, activation=activation_dense,
                        kernel_initializer='TruncatedNormal'))
        model.add(Dropout(dropout_2))

        #Output layer
        model.add(Dense(num_classes, activation='softmax',
                        kernel_initializer='TruncatedNormal'))

        optimizer = SGD(learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        path = str(Path.home())+'/'
        callbacks_list = []
        if checkpoint:
            model_checkpoint = ModelCheckpoint(path+'checkpoint.hdf5', monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
            callbacks_list.append(model_checkpoint)
        if early_stop_callback is not None:
            callbacks_list.append(early_stop_callback)

        if val_X is None:
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1)
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=1)

        return model, history



