#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:40:39 2021

@author: daniel
"""
import os
import tensorflow as tf
#os.environ['PYTHONHASHSEED'] = '0'
#os.environ["TF_DETERMINISTIC_OPS"] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy 
import joblib

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

#SEED_NO = 1909
#import random as python_random
##https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development##
#np.random.seed(SEED_NO), python_random.seed(SEED_NO), tf.random.set_seed(SEED_NO)

from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.initializers import VarianceScaling

from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.losses import categorical_crossentropy, Hinge, SquaredHinge, KLDivergence, LogCosh
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, Conv2D, MaxPool2D, Add, ZeroPadding2D, \
    AveragePooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Lambda, concatenate
from optuna.importance import get_param_importances, FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator
from pyBIA.data_processing import process_class, create_training_set, concat_channels
from pyBIA.data_augmentation import augmentation, resize, plot #smote_oversampling
from pyBIA import optimization


class Classifier:
    """
    Creates and trains a convolutional neural network for binary classification,
    with optional normalization, augmentation, simple cross-validation, and
    convenience utilities for saving, loading, and visualization.

    Parameters
    ----------
    positive_class : ndarray or None, optional
        Training images for the positive class. Accepts (N, H, W) or (N, H, W, C)
        arrays where N is the number of samples, H×W are spatial dimensions,
        and C is the number of channels. Default is None.
    negative_class : ndarray or None, optional
        Training images for the negative class. Accepts (N, H, W) or (N, H, W, C)
        arrays with the same conventions as `positive_class`. Default is None.
    val_positive : ndarray or None, optional
        Optional validation images for the positive class using the same shape
        rules as training data. Default is None.
    val_negative : ndarray or None, optional
        Optional validation images for the negative class using the same shape
        rules as training data. Default is None.
    img_num_channels : int, optional
        Number of channels per image (last dimension). Inferred from 4-D inputs
        when possible; may be set explicitly for legacy compatibility.
        Default is 1.
    clf : {'alexnet','vgg16','resnet18','custom_cnn'}, optional
        Backbone architecture to build and train. Default is 'alexnet'.
    normalize : bool, optional
        If True, min–max normalize each image/channel using `min_pixel` and
        `max_pixel` before training or prediction. Default is False.
    min_pixel : float, optional
        Lower clamp applied during min–max normalization (used only if
        `normalize=True`). Default is 0.
    max_pixel : float or list, optional
        Upper clamp applied during min–max normalization (used only if
        `normalize=True`). If multi-channel, a list may specify per-channel
        maxima. Default is 100.
    epochs : int, optional
        Number of training epochs. If set to 0, the model is constructed but
        not trained. Default is 25.
    patience : int, optional
        Early-stopping patience (epochs) for the monitored `metric`. Default is 5.
    metric : {'loss','binary_accuracy','f1_score','all','val_loss','val_binary_accuracy','val_f1_score'}, optional
        Metric used for monitoring/selection during training/early stopping.
        Default is 'loss'.
    opt_cv : int or None, optional
        If set to an integer K, perform simple K-fold-like training by rotating
        validation blocks (requires `val_positive`/`val_negative`). Default is None.
    augment_data : bool, optional
        If True, apply the configured augmentation pipeline to the training data
        (positive class, and optionally negative). Default is False.
    batch_positive : int, optional
        Augmentation multiplier applied to the positive class (outputs per input).
        Default is 10.
    batch_negative : int, optional
        Augmentation multiplier applied to the negative class (0 disables negative
        augmentation). Default is 1.
    balance : bool, optional
        After augmentation/resizing, trim the larger class to match the smaller.
        Default is True.
    image_size : int, optional
        Target square side length used by augmentation/resize utilities.
        Default is 70.
    shift : int, optional
        Maximum absolute pixel shift applied horizontally and vertically during
        augmentation. Default is 10.
    rotation : bool, optional
        If True, allow random rotations in the full 0–360° range. Default is False.
    horizontal : bool, optional
        If True, allow random horizontal flips in augmentation. Default is False.
    vertical : bool, optional
        If True, allow random vertical flips in augmentation. Default is False.
    mask_size : int or tuple or None, optional
        Side length of random square cutouts applied during augmentation; if a
        tuple (low, high) is given, sizes are sampled uniformly from the range.
        Default is None.
    num_masks : int or tuple or None, optional
        Number of cutouts per image when `mask_size` is set; if a tuple (low, high)
        is given, counts are sampled uniformly from the range. Default is None.
    blend_positive : float, optional
        Blended-image synthesis factor for the positive class (≥1 adds synthetic
        samples, 0 disables blending). Default is 0.
    blending_func : {'mean','max','min','random'}, optional
        Operator used when blending multiple images to synthesize samples.
        Default is 'mean'.
    num_images_to_blend : int, optional
        Number of images combined per synthetic blend operation. Default is 2.
    blend_negative : float, optional
        Blended-image synthesis factor for the negative class (≥1 adds synthetic
        samples, 0 disables blending). Default is 0.
    zoom_range : tuple of (float, float) or None, optional
        Random zoom range specified as (min_zoom, max_zoom). Default is (0.9, 1.1).
    skew_angle : float, optional
        Maximum absolute skew angle in degrees; the actual angle is sampled
        uniformly from [−skew_angle, +skew_angle]. Default is 0.
    batch_size : int, optional
        Mini-batch size used during training. Default is 32.
    optimizer : {'sgd','adam','rmsprop','adadelta',...}, optional
        Optimizer name forwarded to the model builders. Default is 'sgd'.
    lr : float, optional
        Optimizer learning rate. Default is 0.0001.
    momentum : float, optional
        Momentum parameter used by SGD-like optimizers. Default is 0.9.
    decay : float, optional
        Per-epoch learning-rate decay. Default is 0.0.
    nesterov : bool, optional
        If True, use Nesterov momentum with SGD. Default is False.
    rho : float, optional
        Rho parameter for Adadelta/RMSprop optimizers. Default is 0.9.
    beta_1 : float, optional
        Beta1 parameter for Adam-type optimizers. Default is 0.9.
    beta_2 : float, optional
        Beta2 parameter for Adam-type optimizers. Default is 0.999.
    amsgrad : bool, optional
        If True, use the AMSGrad variant of Adam. Default is False.
    conv_init : str, optional
        Kernel initializer for convolutional layers. Default is 'uniform_scaling'.
    dense_init : str, optional
        Kernel initializer for dense layers. Default is 'truncated_normal'.
    activation_conv : str, optional
        Activation function used in convolutional layers. Default is 'relu'.
    activation_dense : str, optional
        Activation function used in dense layers. Default is 'relu'.
    conv_reg : float, optional
        L2 regularization strength applied to convolutional layers. Default is 0.
    dense_reg : float, optional
        L2 regularization strength applied to dense layers. Default is 0.
    padding : {'same','valid'}, optional
        Convolution padding mode used throughout the network. Default is 'same'.
    model_reg : {'batch_norm', None, ...}, optional
        Model-level regularization utility applied to the network. Default is 'batch_norm'.
    verbose : {0,1,2}, optional
        Keras verbosity level (0 = silent, 1 = progress bar, 2 = per-epoch line).
        Default is 2.
    path : str or None, optional
        Base directory for saving/loading artifacts; the home directory is used
        when None. Default is None.
    use_gpu : bool, optional
        If False, disable GPU via the environment variable `CUDA_VISIBLE_DEVICES='-1'`.
        Default is False.

    Attributes
    ----------
    model : keras.Model or list[keras.Model] or None
        Trained model (or list of models if `opt_cv` is used).
    history : keras.callbacks.History or list[History] or None
        Keras history object(s) from training.
    model_train_metrics : ndarray or list[ndarray]
        Stacked training metrics per epoch: columns
        [binary_accuracy, loss, f1_score]. If CV, a list per fold.
    model_val_metrics : ndarray or list[ndarray]
        Same as above but for validation, if validation data was provided.
    path : str or None
        Folder where artifacts are saved/loaded (set by `save()`/`load()`).
    """

    def __init__(
        self, 
        positive_class=None, 
        negative_class=None, 
        val_positive=None, 
        val_negative=None, 
        img_num_channels=1, 
        clf='alexnet',

        normalize=False, 
        min_pixel=0, 
        max_pixel=100, 
        epochs=25, 
        patience=5, 
        metric='loss', 

        opt_cv=None,

        augment_data=False,
        batch_positive=10,
        batch_negative=1,
        balance=True, 
        image_size=70,

        shift=10, 
        rotation=False, 
        horizontal=False, 
        vertical=False, 
        mask_size=None, 
        num_masks=None, 
        blend_positive=0,
        blending_func='mean', 
        num_images_to_blend=2, 
        blend_negative=0, 
        zoom_range=(0.9,1.1), 
        skew_angle=0,

        batch_size=32,
        optimizer='sgd', 
        lr=0.0001, 
        momentum=0.9, 
        decay=0.0, 
        nesterov=False, 
        rho=0.9, 
        beta_1=0.9, 
        beta_2=0.999, 
        amsgrad=False,
        conv_init='uniform_scaling', 
        dense_init='truncated_normal',
        activation_conv='relu', 
        activation_dense='relu', 
        conv_reg=0, 
        dense_reg=0, 
        padding='same', 
        model_reg='batch_norm',
        verbose=2, 
        path=None, 
        use_gpu=False, 
        ):

        # Training data and model
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.val_positive = val_positive
        self.val_negative = val_negative
        self.img_num_channels = img_num_channels
        self.clf = clf

        #Normalization parameters
        self.normalize = normalize
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

        #Training params      
        self.epochs = epochs
        self.patience = patience
        self.metric = metric
        self.opt_cv = opt_cv

        #Augmentation params 
        self.augment_data = augment_data
        self.batch_positive = batch_positive
        self.batch_negative = batch_negative
        self.balance = balance
        self.image_size = image_size

        #Image augmentation procedures
        self.shift = shift
        self.rotation = rotation
        self.horizontal = horizontal
        self.vertical = vertical
        self.mask_size = mask_size
        self.num_masks = num_masks
        self.blending_func = blending_func
        self.num_images_to_blend = num_images_to_blend
        self.blend_negative = blend_negative
        self.zoom_range = zoom_range
        self.skew_angle = skew_angle
        self.blend_positive = blend_positive

        #CNN Model Hyperparameters
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov 
        self.rho = rho
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.amsgrad = amsgrad
        self.conv_init = conv_init
        self.dense_init = dense_init
        self.activation_conv = activation_conv
        self.activation_dense = activation_dense
        self.conv_reg = conv_reg
        self.dense_reg = dense_reg
        self.padding = padding
        self.model_reg = model_reg

        #Verbose following the tf.keras convention
        self.verbose = verbose

        #Path for saving & loading, will start as None and be updated when objects are loaded/saved
        self.path = path

        #Whether to turn off GPU
        self.use_gpu = use_gpu

        if self.use_gpu is False:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

        if self.clf not in ['alexnet', 'vgg16', 'resnet18', 'custom_cnn']:
            raise ValueError('Invalid clf input, options are: "alexnet", "vgg16", "resnet18", or "custom_cnn".')

        if self.positive_class is not None:
            if len(self.positive_class.shape) == 4 and self.img_num_channels != self.positive_class.shape[-1]:
                print('NOTE: Detected {} filters but img_num_channels was set to {}, setting img_numg_channels={}'.format(self.positive_class.shape[-1], self.img_num_channels, self.positive_class.shape[-1]))
                self.img_num_channels = self.positive_class.shape[-1]
            if len(self.positive_class.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                self.positive_class = np.squeeze(self.positive_class)
            if len(self.negative_class.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                self.negative_class = np.squeeze(self.negative_class)
            if self.val_positive is not None:
                if len(self.val_positive.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                    self.val_positive = np.squeeze(self.val_positive)
                if len(self.val_positive.shape) == 2:
                    if self.img_num_channels != 1:
                        raise ValueError('Single image detected as the positive validation data, img_num_channels must be 1!')
                    else:
                        self.val_positive = np.reshape(self.val_positive, (1, self.val_positive.shape[0], self.val_positive.shape[1]))
            if self.val_negative is not None:
                if len(self.val_negative.shape) == 4 and self.img_num_channels == 1: #If it's just one filter convert to 3-D array
                    self.val_negative = np.squeeze(self.val_negative)
                if len(self.val_negative.shape) == 2:
                    if self.img_num_channels != 1:
                        raise ValueError('Single image detected as the negative validation data, img_num_channels must be 1!')
                    else:
                        self.val_negative = np.reshape(self.val_negative, (1, self.val_negative.shape[0], self.val_negative.shape[1]))

        #These will be the model attributes
        self.model = None
        self.history = None 

    def create(self, overwrite_training=False, save_training=False):
        """
        Build and train the configured CNN (optionally with augmentation and CV).

        Models and histories are stored on the instance (`self.model`, `self.history`).
        If `epochs == 0`, the network is constructed but not trained.

        Parameters
        ----------
        overwrite_training : bool, optional
            If True, replace `positive_class`/`negative_class` (and validation sets, if any)
            with the processed arrays actually used for training (after normalization,
            resizing, and augmentation). Default is False.
        save_training : bool, optional
            If True, persist the processed training/validation arrays alongside the model
            artifacts (location determined by `path`). Default is False.

        Returns
        -------
        None
        """

        if self.positive_class is None or self.negative_class is None:
            raise ValueError('No training data found! Input both the positive_class and the negative_class.')
            
        if self.epochs == 0:
            print(); print(f'The epochs parameter is zero, {self.clf} model not trained...')
            return

        else:

            clear_session()

            if self.augment_data:

                if self.img_num_channels == 1:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class), None, None 
                elif self.img_num_channels == 2:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), None 
                elif self.img_num_channels == 3:
                    channel1, channel2, channel3 = copy.deepcopy(self.positive_class[:,:,:,0]), copy.deepcopy(self.positive_class[:,:,:,1]), copy.deepcopy(self.positive_class[:,:,:,2])
                else:
                    raise ValueError('Only three filters are supported!')

                #min_pix, max_pix = self.min_pixel, self.max_pixel

                print()
                print('======= Image Parameters ======')
                print()
                print('Num Augmentations :', self.batch_positive)
                print('Image Size : ', self.image_size)
                print('Min Pixel : ', self.min_pixel)
                print('Max Pixel(s) : ', self.max_pixel)
                print('Vertical/Horizontal Shifts : ', self.shift)
                print('Rotations : ', self.rotation)
                print('Horizontal Flips : ', self.horizontal)
                print('Vertical Flips : ', self.vertical)
                print('Zoom Range : ', self.zoom_range)
                print('Num Masks :', self.num_masks)
                print('Mask Size :', self.mask_size)
                print('Blend Multiplier :', self.blend_positive)
                print('Skew Angle :', self.skew_angle)

                augmented_images = augmentation(
                    channel1=channel1, 
                    channel2=channel2, 
                    channel3=channel3, 
                    batch=self.batch_positive, 
                    width_shift=self.shift, 
                    height_shift=self.shift, 
                    horizontal=self.horizontal, 
                    vertical=self.vertical, 
                    rotation=self.rotation, 
                    image_size=self.image_size, 
                    mask_size=self.mask_size, 
                    num_masks=self.num_masks, 
                    blend_multiplier=self.blend_positive, 
                    blending_func=self.blending_func, 
                    num_images_to_blend=self.num_images_to_blend, 
                    zoom_range=self.zoom_range, 
                    skew_angle=self.skew_angle, 
                    return_stacked=False
                    )

                #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
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

                # Perform same augmentation techniques on other data, batch_negative=1 by default.
                # This is important in case positive class is augmented using masks and/or blending for example, in those cases you want to add this to the negative clas
                # self.batch_negative=0 will avoid this

                if self.batch_negative > 0: 

                    if self.img_num_channels == 1:
                        channel1, channel2, channel3 = copy.deepcopy(self.negative_class), None, None 
                    elif self.img_num_channels == 2:
                        channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), None 
                    elif self.img_num_channels == 3:
                        channel1, channel2, channel3 = copy.deepcopy(self.negative_class[:,:,:,0]), copy.deepcopy(self.negative_class[:,:,:,1]), copy.deepcopy(self.negative_class[:,:,:,2])
                    
                    augmented_images_negative = augmentation(
                        channel1=channel1, 
                        channel2=channel2, 
                        channel3=channel3, 
                        batch=self.batch_negative, 
                        width_shift=self.shift, 
                        height_shift=self.shift, 
                        horizontal=self.horizontal, 
                        vertical=self.vertical, 
                        rotation=self.rotation, 
                        image_size=self.image_size, 
                        mask_size=self.mask_size, 
                        num_masks=self.num_masks, 
                        blend_multiplier=self.blend_negative, 
                        blending_func=self.blending_func, 
                        num_images_to_blend=self.num_images_to_blend, 
                        zoom_range=self.zoom_range, 
                        skew_angle=self.skew_angle,
                        return_stacked=False
                        )
                    
                    #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
                    if self.img_num_channels > 1:
                        class_2=[]
                        if self.img_num_channels == 2:
                            for i in range(len(augmented_images_negative[0])):
                                class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i]))
                        else:
                            for i in range(len(augmented_images_negative[0])):
                                class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i], augmented_images_negative[2][i]))
                        class_2 = np.array(class_2)
                    else:
                        class_2 = augmented_images_negative
                
                else:
                    class_2 = self.negative_class

                # Now ensure the other data is resized accordingly!
                # This is because we recommend that images are larger when input to crop out augmentation effects
                # If the user already ensured the other class is of same size then the following procedure won't do anything to the data 
                if self.img_num_channels == 1:
                    class_2 = resize(class_2, size=self.image_size)
                else:
                    channel1 = resize(class_2[:,:,:,0], size=self.image_size)
                    channel2 = resize(class_2[:,:,:,1], size=self.image_size)
                    if self.img_num_channels == 2:
                        class_2 = concat_channels(channel1, channel2)
                    else:
                        channel3 = resize(class_2[:,:,:,2], size=self.image_size)
                        class_2 = concat_channels(channel1, channel2, channel3)

                if self.val_positive is not None:
                    if self.img_num_channels == 1:
                        val_class_1 = resize(self.val_positive, size=self.image_size)
                    else:
                        val_channel1 = resize(self.val_positive[:,:,:,0], size=self.image_size)
                        val_channel2 = resize(self.val_positive[:,:,:,1], size=self.image_size)
                        if self.img_num_channels == 2:
                            val_class_1 = concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_positive[:,:,:,2], size=self.image_size)
                            val_class_1 = concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_1 = None

                if self.val_negative is not None:
                    if self.img_num_channels == 1:
                        val_class_2 = resize(self.val_negative, size=self.image_size)
                    elif self.img_num_channels > 1:
                        val_channel1 = resize(self.val_negative[:,:,:,0], size=self.image_size)
                        val_channel2 = resize(self.val_negative[:,:,:,1], size=self.image_size)
                        if self.img_num_channels == 2:
                            val_class_2 = concat_channels(val_channel1, val_channel2)
                        else:
                            val_channel3 = resize(self.val_negative[:,:,:,2], size=self.image_size)
                            val_class_2 = concat_channels(val_channel1, val_channel2, val_channel3)
                else:
                    val_class_2 = None
            
                #Balance the class sizes if necessary
                if self.balance:
                    if self.batch_negative > 1:
                        
                        # Must shuffle first if data was augmented!
                        rng = np.random.default_rng(seed=self.SEED_NO)
                        shuffled_indices = rng.permutation(len(class_2))
                        class_2 = class_2[shuffled_indices]

                    class_2 = class_2[:len(class_1)]     

            else:
                class_1, class_2 = self.positive_class, self.negative_class
                val_class_1, val_class_2 = self.val_positive, self.val_negative

           
            if self.opt_cv is not None and self.verbose != 0:
                    print(); print('***********  CV - 1 ***********'); print()

            if self.clf == 'alexnet':

                self.model, self.history = AlexNet(
                    class_1, 
                    class_2, 
                    img_num_channels=self.img_num_channels, 
                    normalize=self.normalize,
                    min_pixel=self.min_pixel, 
                    max_pixel=self.max_pixel, 
                    val_positive=val_class_1, 
                    val_negative=val_class_2, 
                    epochs=self.epochs,
                    batch_size=self.batch_size, 
                    optimizer=self.optimizer, 
                    lr=self.lr, 
                    momentum=self.momentum, 
                    decay=self.decay, 
                    nesterov=self.nesterov, 
                    rho=self.rho,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                    amsgrad=self.amsgrad,
                    conv_init=self.conv_init,
                    dense_init=self.dense_init,
                    activation_conv=self.activation_conv,
                    activation_dense=self.activation_dense,
                    conv_reg=self.conv_reg,
                    dense_reg=self.dense_reg,
                    padding=self.padding,
                    model_reg=self.model_reg,
                    patience=self.patience, 
                    metric=self.metric, 
                    checkpoint=False, 
                    verbose=self.verbose, 
                    save_training_data=save_training, 
                    path=self.path
                    )

            elif self.clf == 'custom_cnn':

                self.model, self.history = custom_model(
                    class_1, 
                    class_2, 
                    img_num_channels=self.img_num_channels, 
                    normalize=self.normalize,
                    min_pixel=self.min_pixel, 
                    max_pixel=self.max_pixel, 
                    val_positive=val_class_1, 
                    val_negative=val_class_2, 
                    epochs=self.epochs,
                    batch_size=self.batch_size, 
                    optimizer=self.optimizer, 
                    lr=self.lr, 
                    momentum=self.momentum, 
                    decay=self.decay, 
                    nesterov=self.nesterov, 
                    rho=self.rho,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                    amsgrad=self.amsgrad,
                    conv_init=self.conv_init,
                    dense_init=self.dense_init,
                    activation_conv=self.activation_conv,
                    activation_dense=self.activation_dense,
                    conv_reg=self.conv_reg,
                    dense_reg=self.dense_reg,
                    padding=self.padding,
                    model_reg=self.model_reg,
                    patience=self.patience, 
                    metric=self.metric, 
                    checkpoint=False, 
                    verbose=self.verbose, 
                    save_training_data=save_training, 
                    path=self.path
                    )

            elif self.clf == 'vgg16':

                self.model, self.history = VGG16(
                    class_1, 
                    class_2, 
                    img_num_channels=self.img_num_channels, 
                    normalize=self.normalize,
                    min_pixel=self.min_pixel, 
                    max_pixel=self.max_pixel, 
                    val_positive=val_class_1, 
                    val_negative=val_class_2, 
                    epochs=self.epochs,
                    batch_size=self.batch_size, 
                    optimizer=self.optimizer, 
                    lr=self.lr, 
                    momentum=self.momentum, 
                    decay=self.decay, 
                    nesterov=self.nesterov, 
                    rho=self.rho,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                    amsgrad=self.amsgrad,
                    conv_init=self.conv_init,
                    dense_init=self.dense_init,
                    activation_conv=self.activation_conv,
                    activation_dense=self.activation_dense,
                    conv_reg=self.conv_reg,
                    dense_reg=self.dense_reg,
                    padding=self.padding,
                    model_reg=self.model_reg,
                    patience=self.patience, 
                    metric=self.metric, 
                    checkpoint=False, 
                    verbose=self.verbose, 
                    save_training_data=save_training, 
                    path=self.path
                    )

            elif self.clf == 'resnet18':

                self.model, self.history = Resnet18(
                    class_1, 
                    class_2, 
                    img_num_channels=self.img_num_channels, 
                    normalize=self.normalize,
                    min_pixel=self.min_pixel, 
                    max_pixel=self.max_pixel, 
                    val_positive=val_class_1, 
                    val_negative=val_class_2, 
                    epochs=self.epochs,
                    batch_size=self.batch_size, 
                    optimizer=self.optimizer, 
                    lr=self.lr, 
                    momentum=self.momentum, 
                    decay=self.decay, 
                    nesterov=self.nesterov, 
                    rho=self.rho,
                    beta_1=self.beta_1,
                    beta_2=self.beta_2,
                    amsgrad=self.amsgrad,
                    conv_init=self.conv_init,
                    dense_init=self.dense_init,
                    activation_conv=self.activation_conv,
                    activation_dense=self.activation_dense,
                    conv_reg=self.conv_reg,
                    dense_reg=self.dense_reg,
                    padding=self.padding,
                    model_reg=self.model_reg,
                    patience=self.patience, 
                    metric=self.metric, 
                    checkpoint=False, 
                    verbose=self.verbose, 
                    save_training_data=save_training, 
                    path=self.path
                    )
        
            #################################

            ##### Cross-Validation Routine - implementation in which the validation data is inserted into the training data with the replacement serving as the new validation#####
            if self.opt_cv is not None:

                models, histories = [], [] #Will be used to append additional models, it opt_cv is enabled

                models.append(self.model); histories.append(self.history) #Appending the already created first model & history

                if self.val_positive is None and self.val_negative is None:
                    raise ValueError('CNN cross-validation is only supported if validation data is input.')
                if self.val_positive is not None:
                    if len(self.positive_class) / len(self.val_positive) < self.opt_cv-1:
                        raise ValueError('Cannot evenly partition the positive training/validation data, refer to the pyBIA API documentation for instructions on how to use the opt_cv parameter.')
                #if self.val_negative is not None:
                #    if len(self.negative_class) / len(self.val_negative) < self.opt_cv-1:
                #        raise ValueError('Cannot evenly partition the negative training/validation data, refer to the pyBIA API documentation for instructions on how to use the opt_cv parameter.')
                
                #The first model (therefore the first "fold") already ran, therefore sutbract 1      
                for k in range(self.opt_cv-1):        

                    #Make deep copies to avoid overwriting arrays
                    class_1, class_2 = copy.deepcopy(self.positive_class), copy.deepcopy(self.negative_class)
                    val_class_1, val_class_2 = copy.deepcopy(self.val_positive), copy.deepcopy(self.val_negative)

                    #Sort the new data samples, no random shuffling, just a linear sequence
                    if val_class_1 is not None:
                        val_hold_1 = copy.deepcopy(class_1[k*len(val_class_1):len(val_class_1)*(k+1)]) #The new positive validation data
                        class_1[k*len(val_class_1):len(val_class_1)*(k+1)] = copy.deepcopy(val_class_1) #The new class_1, copying to avoid linkage between arrays
                        val_class_1 = val_hold_1 
                    #if val_class_2 is not None:
                    #    val_hold_2 = copy.deepcopy(class_2[k*len(val_class_2):len(val_class_2)*(k+1)]) #The new validation data
                    #    class_2[k*len(val_class_2):len(val_class_2)*(k+1)] = copy.deepcopy(val_class_2) #The new class_2, copying to avoid linkage between arrays
                    #    val_class_2 = val_hold_2 

                    if self.augment_data:

                        if self.img_num_channels == 1:
                            channel1, channel2, channel3 = copy.deepcopy(class_1), None, None 
                        elif self.img_num_channels == 2:
                            channel1, channel2, channel3 = copy.deepcopy(class_1[:,:,:,0]), copy.deepcopy(class_1[:,:,:,1]), None 
                        else:
                            channel1, channel2, channel3 = copy.deepcopy(class_1[:,:,:,0]), copy.deepcopy(class_1[:,:,:,1]), copy.deepcopy(class_1[:,:,:,2])

                        augmented_images = augmentation(
                            channel1=channel1, 
                            channel2=channel2, 
                            channel3=channel3, 
                            batch=self.batch_positive, 
                            width_shift=self.shift, 
                            height_shift=self.shift, 
                            horizontal=self.horizontal, 
                            vertical=self.vertical, 
                            rotation=self.rotation, 
                            image_size=self.image_size, 
                            mask_size=self.mask_size, 
                            num_masks=self.num_masks, 
                            blend_multiplier=self.blend_positive, 
                            blending_func=self.blending_func, 
                            num_images_to_blend=self.num_images_to_blend, 
                            zoom_range=self.zoom_range, 
                            skew_angle=self.skew_angle
                            )

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

                        if self.batch_negative > 0:

                            #Perform same augmentation techniques on negative class data, batch_negative=1 by default
                            if self.img_num_channels == 1:
                                channel1, channel2, channel3 = copy.deepcopy(class_2), None, None 
                            elif self.img_num_channels == 2:
                                channel1, channel2, channel3 = copy.deepcopy(class_2[:,:,:,0]), copy.deepcopy(class_2[:,:,:,1]), None 
                            elif self.img_num_channels == 3:
                                channel1, channel2, channel3 = copy.deepcopy(class_2[:,:,:,0]), copy.deepcopy(class_2[:,:,:,1]), copy.deepcopy(class_2[:,:,:,2])
                            
                            augmented_images_negative = augmentation(
                                channel1=channel1, 
                                channel2=channel2, 
                                channel3=channel3, 
                                batch=self.batch_negative, 
                                width_shift=self.shift, 
                                height_shift=self.shift, 
                                horizontal=self.horizontal, 
                                vertical=self.vertical, 
                                rotation=self.rotation, 
                                image_size=self.image_size, 
                                mask_size=self.mask_size, 
                                num_masks=self.num_masks, 
                                blend_multiplier=self.blend_negative, 
                                blending_func=self.blending_func, 
                                num_images_to_blend=self.num_images_to_blend, 
                                zoom_range=self.zoom_range, 
                                skew_angle=self.skew_angle)

                            #The augmentation routine returns an output for each filter, e.g. 3 outputs for RGB
                            if self.img_num_channels > 1:
                                class_2=[]
                                if self.img_num_channels == 2:
                                    for i in range(len(augmented_images_negative[0])):
                                        class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i]))
                                else:
                                    for i in range(len(augmented_images_negative[0])):
                                        class_2.append(concat_channels(augmented_images_negative[0][i], augmented_images_negative[1][i], augmented_images_negative[2][i]))
                                class_2 = np.array(class_2)
                            else:
                                class_2 = augmented_images_negative

                        else:
                            class_2 = self.negative_class


                        # Now ensure the other data is resized accordingly!
                        # This is because we recommend that images are larger when input to crop out augmentation effects
                        # If the user already ensured the other class is of same size then the following procedure won't do anything to the data 
                        if self.img_num_channels == 1:
                            class_2 = resize(class_2, size=self.image_size)
                        else:
                            channel1 = resize(class_2[:,:,:,0], size=self.image_size)
                            channel2 = resize(class_2[:,:,:,1], size=self.image_size)
                            if self.img_num_channels == 2:
                                class_2 = concat_channels(channel1, channel2)
                            else:
                                channel3 = resize(class_2[:,:,:,2], size=self.image_size)
                                class_2 = concat_channels(channel1, channel2, channel3)

                        if val_class_1 is not None:
                            if self.img_num_channels == 1:
                                val_class_1 = resize(val_class_1, size=self.image_size)
                            else:
                                val_channel1 = resize(val_class_1[:,:,:,0], size=self.image_size)
                                val_channel2 = resize(val_class_1[:,:,:,1], size=self.image_size)
                                if self.img_num_channels == 2:
                                    val_class_1 = concat_channels(val_channel1, val_channel2)
                                else:
                                    val_channel3 = resize(val_class_1[:,:,:,2], size=self.image_size)
                                    val_class_1 = concat_channels(val_channel1, val_channel2, val_channel3)

                        if val_class_2 is not None:
                            if self.img_num_channels == 1:
                                val_class_2 = resize(val_class_2, size=self.image_size)
                            elif self.img_num_channels > 1:
                                val_channel1 = resize(val_class_2[:,:,:,0], size=self.image_size)
                                val_channel2 = resize(val_class_2[:,:,:,1], size=self.image_size)
                                if self.img_num_channels == 2:
                                    val_class_2 = concat_channels(val_channel1, val_channel2)
                                else:
                                    val_channel3 = resize(val_class_2[:,:,:,2], size=self.image_size)
                                    val_class_2 = concat_channels(val_channel1, val_channel2, val_channel3)

                         #Balance the class sizes if necessary
                        if self.balance:

                            if self.batch_negative > 1: 

                                # Must shuffle first if data was augmented!
                                rng = np.random.default_rng(seed=self.SEED_NO)
                                shuffled_indices = rng.permutation(len(class_2))
                                class_2 = class_2[shuffled_indices]

                            class_2 = class_2[:len(class_1)]   

                    if self.verbose != 0:
                        print(); print('***********  CV - {} ***********'.format(k+2)); print()

                    clear_session()

                    if self.clf == 'alexnet':

                        model, history = AlexNet(
                            class_1, 
                            class_2, 
                            img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, 
                            min_pixel=self.min_pixel, 
                            max_pixel=self.max_pixel, 
                            val_positive=val_class_1, 
                            val_negative=val_class_2, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            optimizer=self.optimizer, 
                            lr=self.lr, 
                            momentum=self.momentum, 
                            decay=self.decay, 
                            nesterov=self.nesterov, 
                            rho=self.rho,
                            beta_1=self.beta_1,
                            beta_2=self.beta_2,
                            amsgrad=self.amsgrad,
                            conv_init=self.conv_init,
                            dense_init=self.dense_init,
                            activation_conv=self.activation_conv,
                            activation_dense=self.activation_dense,
                            conv_reg=self.conv_reg,
                            dense_reg=self.dense_reg,
                            padding=self.padding,
                            model_reg=self.model_reg,
                            patience=self.patience, 
                            metric=self.metric, 
                            checkpoint=False, 
                            verbose=self.verbose, #self.verbose, when doing cross-validation no need to always print model architecture 
                            save_training_data=save_training, 
                            path=self.path
                            )

                    elif self.clf == 'custom_cnn':

                        model, history = custom_model(
                            class_1, 
                            class_2, 
                            img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, 
                            min_pixel=self.min_pixel, 
                            max_pixel=self.max_pixel, 
                            val_positive=val_class_1, 
                            val_negative=val_class_2, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            optimizer=self.optimizer, 
                            lr=self.lr, 
                            momentum=self.momentum, 
                            decay=self.decay, 
                            nesterov=self.nesterov, 
                            rho=self.rho,
                            beta_1=self.beta_1,
                            beta_2=self.beta_2,
                            amsgrad=self.amsgrad,
                            conv_init=self.conv_init,
                            dense_init=self.dense_init,
                            activation_conv=self.activation_conv,
                            activation_dense=self.activation_dense,
                            conv_reg=self.conv_reg,
                            dense_reg=self.dense_reg,
                            padding=self.padding,
                            model_reg=self.model_reg,
                            patience=self.patience, 
                            metric=self.metric, 
                            checkpoint=False, 
                            verbose=self.verbose, #self.verbose, when doing cross-validation no need to always print model architecture
                            save_training_data=save_training, 
                            path=self.path
                            )

                    elif self.clf == 'vgg16':

                        model, history = VGG16(
                            class_1, 
                            class_2, 
                            img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, 
                            min_pixel=self.min_pixel, 
                            max_pixel=self.max_pixel, 
                            val_positive=val_class_1, 
                            val_negative=val_class_2, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            optimizer=self.optimizer, 
                            lr=self.lr, 
                            momentum=self.momentum, 
                            decay=self.decay, 
                            nesterov=self.nesterov, 
                            rho=self.rho,
                            beta_1=self.beta_1,
                            beta_2=self.beta_2,
                            amsgrad=self.amsgrad,
                            conv_init=self.conv_init,
                            dense_init=self.dense_init,
                            activation_conv=self.activation_conv,
                            activation_dense=self.activation_dense,
                            conv_reg=self.conv_reg,
                            dense_reg=self.dense_reg,
                            padding=self.padding,
                            model_reg=self.model_reg,
                            patience=self.patience, 
                            metric=self.metric, 
                            checkpoint=False, 
                            verbose=self.verbose, #self.verbose, when doing cross-validation no need to always print model architecture
                            save_training_data=save_training, 
                            path=self.path
                            )

                    elif self.clf == 'resnet18':
                        model, history = Resnet18(
                            class_1, 
                            class_2, 
                            img_num_channels=self.img_num_channels, 
                            normalize=self.normalize, 
                            min_pixel=self.min_pixel, 
                            max_pixel=self.max_pixel, 
                            val_positive=val_class_1, 
                            val_negative=val_class_2, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            optimizer=self.optimizer, 
                            lr=self.lr, 
                            momentum=self.momentum, 
                            decay=self.decay, 
                            nesterov=self.nesterov, 
                            rho=self.rho,
                            beta_1=self.beta_1,
                            beta_2=self.beta_2,
                            amsgrad=self.amsgrad,
                            conv_init=self.conv_init,
                            dense_init=self.dense_init,
                            activation_conv=self.activation_conv,
                            activation_dense=self.activation_dense,
                            conv_reg=self.conv_reg,
                            dense_reg=self.dense_reg,
                            padding=self.padding,
                            model_reg=self.model_reg,
                            patience=self.patience, 
                            metric=self.metric, 
                            checkpoint=False, 
                            verbose=self.verbose, #self.verbose, when doing cross-validation no need to always print model architecture 
                            save_training_data=save_training, 
                            path=self.path
                            )

                    models.append(model), histories.append(history)

                    try:
                        if np.isfinite(history.history['loss'][-1]) is False:
                            print(); print(f"NOTE: Training failed during fold {k} due to numerical instability!")
                    except Exception as e:    
                        print(); print(f"ERROR: Training failed during fold {k} due to error: {e}!")
                        return

            #################################
            if self.opt_cv is None:
                self.model_train_metrics = np.c_[self.history.history['binary_accuracy'], self.history.history['loss'], self.history.history['f1_score']]
                if self.val_positive is not None:
                    self.model_val_metrics = np.c_[self.history.history['val_binary_accuracy'], self.history.history['val_loss'], self.history.history['val_f1_score']]
                print('Complete! To save the final model and optimization results, call the save() method.') 
            else:
                self.model, self.history = models, histories
                self.model_train_metrics = [] 
                for i in range(100): #If more than 100 CVs then this will break 
                    try:
                        model_train_metrics = np.c_[self.history[i].history['binary_accuracy'], self.history[i].history['loss'], self.history[i].history['f1_score']]
                        self.model_train_metrics.append(model_train_metrics)
                    except:
                        break

                if self.val_positive is not None:
                    self.model_val_metrics = []
                    for i in range(100): #If more than 100 CVs then this will break 
                        try:
                            model_val_metrics = np.c_[self.history[i].history['val_binary_accuracy'], self.history[i].history['val_loss'], self.history[i].history['val_f1_score']]
                            self.model_val_metrics.append(model_val_metrics)
                        except:
                            break

                print('Complete!'); print('NOTE: Cross-validation was enabled, therefore the model and history class attribute are lists containing all. To save, call the save() method.') 

            if overwrite_training:
                if self.normalize:
                    #
                    class_1 = process_class(class_1, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)
                    class_2 = process_class(class_2, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)
                    if val_class_1 is not None:
                        val_class_1 = process_class(val_class_1, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)
                    if val_class_2 is not None:
                        val_class_2 = process_class(val_class_2, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)

                self.positive_class, self.negative_class, self.val_positive, self.val_negative = class_1, class_2, val_class_1, val_class_2

            return

    def save(self, dirname=None, overwrite=False):
        """
        Save the trained model(s), metrics, and class attributes to disk.

        Creates a folder named 'pyBIA_cnn_model' under the base directory (`path` or
        home directory). If multiple CV models exist, each is saved separately.

        Parameters
        ----------
        dirname : str or None, optional
            Optional subdirectory created beneath the base path before saving
            (e.g., to group experiments). If the subdirectory already exists,
            an error is raised unless handled by the caller. Default is None.
        overwrite : bool, optional
            If True, delete any existing 'pyBIA_cnn_model' folder at the target
            location and recreate it to avoid duplicates. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the destination folder already exists and `overwrite` is False.
        """

        if self.model is None:
            print('The model has not been created!')

        path = str(Path.home()) if self.path is None else self.path
        path += '/' if path[-1] != '/' else ''

        if dirname is not None:
            if dirname[-1] != '/':
                dirname+='/'
            path = path+dirname
            try:
                os.makedirs(path)
            except FileExistsError:
                raise ValueError('The dirname folder already exists!')

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
            if isinstance(self.model, list) is False:
                save_model(self.model, path+'Keras_Model.h5')#,  custom_objects={'f1_score': f1_score})
                np.savetxt(path+'model_train_metrics', np.c_[self.history.history['binary_accuracy'], self.history.history['loss'], self.history.history['f1_score']], header='binary_accuracy\tloss\tf1_score')
                if self.val_positive is not None:
                    np.savetxt(path+'model_val_metrics', np.c_[self.history.history['val_binary_accuracy'], self.history.history['val_loss'], self.history.history['val_f1_score']], header='val_binary_accuracy\tval_loss\tval_f1_score')
            else:
                for counter in range(len(self.model)):
                    save_model(self.model[counter], path+'Keras_Model_CV_'+str(counter+1)+'.h5')#,  custom_objects={'f1_score': f1_score})
                    np.savetxt(path+'model_train_metrics_CV_'+str(counter+1), np.c_[self.history[counter].history['binary_accuracy'], self.history[counter].history['loss'], self.history[counter].history['f1_score']], header='binary_accuracy\tloss\tf1_score')
                    if self.val_positive is not None:
                        np.savetxt(path+'model_val_metrics_CV_'+str(counter+1), np.c_[self.history[counter].history['val_binary_accuracy'], self.history[counter].history['val_loss'], self.history[counter].history['val_f1_score']], header='val_binary_accuracy\tval_loss\tval_f1_score')


        try:
            #Save all class attributes except the ones that are generated during the routine, as these are saved above
            exclude_attrs = ['positive_class', 'negative_class', 'val_positive', 
                             'val_negative', 'model', 'history']
            attrs_dict = {attr: getattr(self, attr) for attr in dir(self) 
                          if not callable(getattr(self, attr)) and 
                          not attr.startswith("__") and 
                          attr not in exclude_attrs}
            joblib.dump(attrs_dict, path + 'class_attributes.pkl')
            print('Succesfully saved all class attributes!')
        except Exception as e:
            print(f"Could not save all class attributes to {path} due to error: {e}")

        print('Files saved in: {}'.format(path))
        self.path = path

        return 

    def load(self, path=None, load_training_data=False):
        """
        Load a saved model (or CV models), metrics, and class attributes from disk.

        Looks for a 'pyBIA_cnn_model' folder under `path` (or the home directory if
        `path` is None). Optionally restores the saved training/validation arrays.

        Parameters
        ----------
        path : str or None, optional
            Base directory that contains the 'pyBIA_cnn_model' folder to load from.
            If None, the home directory is used. Default is None.
        load_training_data : bool, optional
            If True, also load the saved `positive_class`, `negative_class`, and
            optional validation arrays (when present). Default is False.

        Returns
        -------
        None
        """

        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        path += 'pyBIA_cnn_model/'

        try:
            attrs_dict = joblib.load(path + 'class_attributes.pkl')
            for attr, value in attrs_dict.items():
                setattr(self, attr, value)
            class_attributes = ', class_attributes'
        except:
            class_attributes = ''

        try:
            self.model = load_model(path+'Keras_Model.h5', compile=False) #custom_objects={'f1_score': f1_score, 'loss': loss})
            model = 'model'
        except:
            try:
                self.model = []
                for i in range(1, 101): #If more than 100 CVs then this will break 
                    try:
                        model = load_model(path+'Keras_Model_CV_'+str(i)+'.h5', compile=False) #custom_objects={'f1_score': f1_score, 'loss': loss})
                        self.model.append(model)
                    except:
                        break

                if len(self.model) >= 1:
                    model = 'models'
                else:
                    print('Could not load models!')
                    model = ''
            except:
                print('Could not load model!')
                model = ''

        try:
            self.model_train_metrics = np.loadtxt(path+'model_train_metrics')
            train_metrics = ', training_history'
        except:
            try:
                self.model_train_metrics = [] 
                for i in range(1, 101): #If more than 100 CVs then this will break 
                    try:
                        model_train_metrics = np.loadtxt(path+'model_train_metrics_CV_'+str(i)) 
                        self.model_train_metrics.append(model_train_metrics)
                    except:
                        continue

                if len(self.model_train_metrics) >= 1:
                    train_metrics = ', training_histories'
                else:
                    print('Could not load training histories!')
                    train_metrics = ''
            except:
                print('Could not load training history!')
                train_metrics = ''

        try:
            self.model_val_metrics = np.loadtxt(path+'model_val_metrics')
            val_metrics = ', val_training_history'
        except:
            try:
                self.model_val_metrics = []
                for i in range(1, 101): #If more than 100 CVs then this will break 
                    try:
                        model_val_metrics = np.loadtxt(path+'model_val_metrics_CV_'+str(i)) 
                        self.model_val_metrics.append(model_val_metrics)
                    except:
                        continue

                if len(self.model_val_metrics) >= 1:
                    val_metrics = ', val_training_histories'
                else:
                    print('Could not load validation training histories!')
                    val_metrics = ''
            except:
                print('Could not load training history!')
                val_metrics = ''


        if load_training_data:
            if self.opt_cv is None:
                print('IMPORTANT: If re-creating the model with loaded data, set opt_aug=False and normalize=False to avoid re-augmenting and re-normalizing the loaded data!')
            else:
                print('IMPORTANT: If re-creating the model with loaded data, set opt_aug=False and normalize=False to avoid re-augmenting and re-normalizing the loaded data! Also, set opt_cv=None if the training data has been augmented!')
            
            try:
                self.positive_class = np.load(path+'class_1.npy')
                positive_class = ', positive_class'
            except:
                positive_class = ''

            try:
                self.negative_class = np.load(path+'class_2.npy')
                negative_class = ', negative_class'
            except:
                negative_class = ''

            try:
                self.val_positive = np.load(path+'val_class_1.npy')
                val_positive = ', val_positive'
            except:
                val_positive = ''

            try:
                self.val_negative = np.load(path+'val_class_2.npy')
                val_negative = ', val_negative'
            except:
                val_negative = ''

            print('Successfully loaded the following: {}{}{}{}{}{}{}{}'.format(model, train_metrics, val_metrics, class_attributes, positive_class, negative_class, val_positive, val_negative))
        else:
            print('Successfully loaded the following: {}{}{}{}'.format(model, train_metrics, val_metrics, class_attributes))

        self.path = path

        return

    def predict(self, data, target='LAB', return_proba=False, cv_model=0):
        """
        Predict class labels for new images using the trained CNN.

        Images are preprocessed using the current normalization settings and resized
        to the model’s required input size when necessary.

        Parameters
        ----------
        data : ndarray
            Input images as (N, H, W) for single-channel or (N, H, W, C) for multi-channel.
            A single image may be passed as (H, W) or (H, W, C) and will be promoted.
        target : str, optional
            String label used for the positive class when returning class names.
            Default is 'LAB'.
        return_proba : bool, optional
            If True, also return predicted probabilities for the positive class.
            Default is False.
        cv_model : int or {'all'}, optional
            Index of the CV model to use when multiple models were trained, or 'all'
            to average probabilities across all models. Default is 0.

        Returns
        -------
        ndarray
            If `return_proba` is False, an array of predicted class strings with shape (N,).
            If `return_proba` is True, an array of shape (N, 2) with columns
            [predicted_label, probability_for_target].

        Raises
        ------
        ValueError
            If no trained model is available or if inputs are incompatible with the model.
        """

        data = process_class(data, normalize=self.normalize, min_pixel=self.min_pixel, max_pixel=self.max_pixel, img_num_channels=self.img_num_channels)
        if self.normalize:
            data[data > 1] = 1; data[data < 0] = 0

        model = self.model[0] if isinstance(self.model, list) else self.model 
        image_size = model.input_shape[1] #layers[0].input_shape[1:][0]

        if data.shape[1] != image_size:
            if data.shape[1] < image_size:
                raise ValueError('Model requires images of size {}, but the input images are size {}!'.format(image_size, data.shape[1]))
            print('Incorrect image size, the model requires size {}, resizing...'.format(image_size))
            data = resize(data, image_size)
    

        if isinstance(self.model, list) is False or isinstance(cv_model, int):

            model = self.model[cv_model] if isinstance(self.model, list) else self.model
            predictions = model.predict(data)

            output, probas = [], [] 
            for i in range(len(predictions)):
                prediction = target if predictions[i] >= 0.5 else 'OTHER'
                probas.append(predictions[i])
                output.append(prediction)

            output = np.c_[output, probas] if return_proba else np.array(output)
            
        else: #cv_model='all' 

            model_outputs, model_probas = [], []
            for __model__ in self.model:

                predictions = __model__.predict(data)

                output, probas = [], []                 
                for i in range(len(predictions)):
                    prediction = target if predictions[i] >= 0.5 else 'OTHER'
                    probas.append(predictions[i])
                    output.append(prediction)

                model_outputs.append(output); model_probas.append(probas)

            average_output, average_proba = [], [] 
            for j in range(len(model_outputs[0])):
                column = [model_outputs[i][j] for i in range(len(model_outputs))]
                avg_proba = np.mean([model_probas[i][j] for i in range(len(model_probas))])
                #avg_output = target if column.count(target) >= column.count('OTHER') else 'OTHER'
                avg_output = target if avg_proba >= 0.5 else 'OTHER'

                average_output.append(avg_output); average_proba.append(avg_proba)

            output = np.c_[average_output, average_proba] if return_proba else np.array(average_output)
            
        return output

    def augment_positive(
        self, 
        batch=1, 
        width_shift=0, 
        height_shift=0, 
        horizontal=False, 
        vertical=False, 
        rotation=False, 
        fill='nearest', 
        image_size=None, 
        zoom_range=None, 
        mask_size=None, 
        num_masks=None, 
        blend_multiplier=0, 
        blending_func='mean', 
        num_images_to_blend=2, 
        skew_angle=0
        ):
        """
        Apply the configured augmentation pipeline to the positive class and replace it.

        Parameters
        ----------
        batch : int, optional
            Number of augmented outputs to create per input image. Default is 1.
        width_shift : int, optional
            Maximum horizontal pixel shift applied uniformly at random. Default is 0.
        height_shift : int, optional
            Maximum vertical pixel shift applied uniformly at random. Default is 0.
        horizontal : bool, optional
            If True, allow random horizontal flips. Default is False.
        vertical : bool, optional
            If True, allow random vertical flips. Default is False.
        rotation : bool, optional
            If True, allow random rotations in the full 0–360° range. Default is False.
        fill : {'constant','nearest','reflect','wrap'}, optional
            Fill mode for pixels introduced by rotations/shifts. Default is 'nearest'.
        image_size : int or None, optional
            Target square size after augmentation; if None, keep original size.
            Default is None.
        zoom_range : tuple of (float, float) or None, optional
            Random zoom range given as (min_zoom, max_zoom). Default is None.
        mask_size : int or None, optional
            Side length of each random square cutout; if None, disable cutouts.
            Default is None.
        num_masks : int or None, optional
            Number of cutouts applied per image when `mask_size` is set. Default is None.
        blend_multiplier : float, optional
            Synthetic blending factor (≥1 adds blended samples, 0 disables blending).
            Default is 0.
        blending_func : {'mean','max','min','random'}, optional
            Operator used when blending multiple images. Default is 'mean'.
        num_images_to_blend : int, optional
            Number of images combined per synthetic blend operation. Default is 2.
        skew_angle : float, optional
            Maximum absolute skew angle in degrees; sampled uniformly from
            [−skew_angle, +skew_angle]. Default is 0.

        Returns
        -------
        None
        """

        #The augmentation function takes in each channel as individual inputs
        if self.img_num_channels == 1:
            channel1, channel2, channel3 = self.positive_class, None, None 
        elif self.img_num_channels == 2:
            channel1, channel2, channel3 = self.positive_class[:,:,:,0], self.positive_class[:,:,:,1], None 
        elif self.img_num_channels == 3:
            channel1, channel2, channel3 = self.positive_class[:,:,:,0], self.positive_class[:,:,:,1], self.positive_class[:,:,:,2]
        
        self.positive_class = augmentation(channel1, channel2, channel3, batch=batch, width_shift=width_shift, height_shift=height_shift, 
            horizontal=horizontal, vertical=vertical, rotation=rotation, fill=fill, image_size=image_size, zoom_range=zoom_range, 
            mask_size=mask_size, num_masks=num_masks, blend_multiplier=blend_multiplier, blending_func=blending_func, num_images_to_blend=num_images_to_blend, 
            skew_angle=skew_angle, return_stacked=True)

        return 

    def augment_negative(
        self, 
        batch=1, 
        width_shift=0, 
        height_shift=0, 
        horizontal=False, 
        vertical=False, 
        rotation=False, 
        fill='nearest', 
        image_size=None, 
        zoom_range=None, 
        mask_size=None, 
        num_masks=None, 
        blend_multiplier=0, 
        blending_func='mean', 
        num_images_to_blend=2, 
        skew_angle=0
        ):
        """
        Apply the configured augmentation pipeline to the negative class and replace it.

        Parameters
        ----------
        batch : int, optional
            Number of augmented outputs to create per input image. Default is 1.
        width_shift : int, optional
            Maximum horizontal pixel shift applied uniformly at random. Default is 0.
        height_shift : int, optional
            Maximum vertical pixel shift applied uniformly at random. Default is 0.
        horizontal : bool, optional
            If True, allow random horizontal flips. Default is False.
        vertical : bool, optional
            If True, allow random vertical flips. Default is False.
        rotation : bool, optional
            If True, allow random rotations in the full 0–360° range. Default is False.
        fill : {'constant','nearest','reflect','wrap'}, optional
            Fill mode for pixels introduced by rotations/shifts. Default is 'nearest'.
        image_size : int or None, optional
            Target square size after augmentation; if None, keep original size.
            Default is None.
        zoom_range : tuple of (float, float) or None, optional
            Random zoom range given as (min_zoom, max_zoom). Default is None.
        mask_size : int or None, optional
            Side length of each random square cutout; if None, disable cutouts.
            Default is None.
        num_masks : int or None, optional
            Number of cutouts applied per image when `mask_size` is set. Default is None.
        blend_multiplier : float, optional
            Synthetic blending factor (≥1 adds blended samples, 0 disables blending).
            Default is 0.
        blending_func : {'mean','max','min','random'}, optional
            Operator used when blending multiple images. Default is 'mean'.
        num_images_to_blend : int, optional
            Number of images combined per synthetic blend operation. Default is 2.
        skew_angle : float, optional
            Maximum absolute skew angle in degrees; sampled uniformly from
            [−skew_angle, +skew_angle]. Default is 0.

        Returns
        -------
        None
        """

        #The augmentation function takes in each channel as individual inputs
        if self.img_num_channels == 1:
            channel1, channel2, channel3 = self.negative_class, None, None 
        elif self.img_num_channels == 2:
            channel1, channel2, channel3 = self.negative_class[:,:,:,0], self.negative_class[:,:,:,1], None 
        elif self.img_num_channels == 3:
            channel1, channel2, channel3 = self.negative_class[:,:,:,0], self.negative_class[:,:,:,1], self.negative_class[:,:,:,2]
        
        self.negative_class = augmentation(channel1, channel2, channel3, batch=batch, width_shift=width_shift, height_shift=height_shift, 
            horizontal=horizontal, vertical=vertical, rotation=rotation, fill=fill, image_size=image_size, zoom_range=zoom_range, 
            mask_size=mask_size, num_masks=num_masks, blend_multiplier=blend_multiplier, blending_func=blending_func, num_images_to_blend=num_images_to_blend, 
            skew_angle=skew_angle, return_stacked=True)

        return 

    def plot_tsne(
        self, 
        legend_loc='upper center', 
        title='Feature Parameter Space', 
        savefig=False):
        """
        Plot a t-SNE projection of images (train and optional validation).

        Data are flattened per image and embedded into 2D using sklearn’s TSNE.

        Parameters
        ----------
        legend_loc : str, optional
            Matplotlib legend location string (e.g., 'upper center'). Default is 'upper center'.
        title : str, optional
            Figure title displayed above the plot. Default is 'Feature Parameter Space'.
        savefig : bool, optional
            If True, save the figure to 'Images_tSNE_Projection.png' instead of showing it.
            Default is False.

        Returns
        -------
        None

        Notes
        -----
        Data should be normalized prior to plotting for meaningful distances
        (see the `normalize` option during training).
        """


        if not (hasattr(self, 'positive_class') and hasattr(self, 'negative_class')):
            raise ValueError('The training data is missing! Make sure the positive_class and negative_class are input.')

        #Reshape if 3D array (single-band) -- need 4D array first.
        if len(self.positive_class.shape) == 3:
            positive_class = np.reshape(self.positive_class, (self.positive_class.shape[0], self.positive_class.shape[1], self.positive_class.shape[2], 1))
            negative_class = np.reshape(self.negative_class, (self.negative_class.shape[0], self.negative_class.shape[1], self.negative_class.shape[2], 1))
            data = np.r_[positive_class, negative_class]
            data_y = np.r_[['LAB Train']*len(positive_class),['OTHER Train']*len(negative_class)]
            if self.val_positive is not None:
                val_positive = np.reshape(self.val_positive, (self.val_positive.shape[0], self.val_positive.shape[1], self.val_positive.shape[2], 1))
            if self.val_negative is not None:
                val_negative = np.reshape(self.val_negative, (self.val_negative.shape[0], self.val_negative.shape[1], self.val_negative.shape[2], 1))
            if self.val_positive is not None and self.val_negative is not None:
                val_data = np.r_[val_positive, val_negative]
                val_data_y = np.r_[['LAB Val']*len(val_positive),['OTHER Val']*len(val_negative)]
            elif self.val_positive is not None and self.val_negative is None:
                val_data = val_positive
                val_data_y = np.r_[['LAB Val']*len(val_data)]
            elif self.val_positive is None and self.val_negative is not None:
                val_data = val_negative
                val_data_y = np.r_[['OTHER Val']*len(val_data)]
            else:
                val_data = val_data_y = None 
        else:
            data = np.r_[self.positive_class, self.negative_class]
            data_y = np.r_[['LAB Train']*len(self.positive_class),['OTHER Train']*len(self.negative_class)]
            if self.val_positive is not None and self.val_negative is not None:
                val_data = np.r_[self.val_positive, self.val_negative]
                val_data_y = np.r_[['LAB Val']*len(self.val_positive),['OTHER Val']*len(self.val_negative)]
            elif self.val_positive is not None and self.val_negative is None:
                val_data = self.val_positive
                val_data_y = np.r_[['LAB Val']*len(val_data)]
            elif self.val_positive is None and self.val_negative is not None:
                val_data = self.val_negative
                val_data_y = np.r_[['OTHER Val']*len(val_data)]
            else:
                val_data = val_data_y = None 

        if val_data is not None:
            data = np.r_[data, val_data]
            data_y = np.r_[data_y, val_data_y]

        #Assuming img_array is a 4D array of shape, which is the standard input for CNN models
        num_images, image_size = data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]
        #Flatten each image to a 1D array
        data_x = np.reshape(data, (num_images, image_size))
        #print(flattened_images.shape)
        #Concatenate the flattened images along the first axis, this can now be input into ensemble algorithms like XGBoost
        #data_x = np.concatenate(flattened_images, axis=0)

        if len(data_x) > 5e3:
            method = 'barnes_hut' #Scales with O(N)
        else:
            method = 'exact' #Scales with O(N^2)
        print(data_x.shape)
        feats = TSNE(n_components=2, method=method, learning_rate=1000, 
            perplexity=200, init='random').fit_transform(data_x)
        x, y = feats[:,0], feats[:,1]

        markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

        feats = np.unique(data_y)

        for count, feat in enumerate(feats):
            marker = markers[count % len(markers)]  # Wrap around the markers list
            color_val = color[count % len(color)]  # Wrap around the color list
            mask = np.where(data_y == feat)[0]
            plt.scatter(x[mask], y[mask], marker=marker, c=color_val, label=str(feat), alpha=0.44)
        """
        markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
        #color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c']
        color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

        feats = np.unique(data_y) 

        for count, feat in enumerate(feats):
            if count+1 > len(markers):
                count = -1
            mask = np.where(data_y == feat)[0]
            plt.scatter(x[mask], y[mask], marker=markers[count], c=color[count], label=str(feat), alpha=0.44)
        """
        plt.legend(loc=legend_loc, ncol=len(np.unique(data_y)), frameon=False, handlelength=2)#prop={'size': 14}
        plt.title(title)
        plt.xticks()
        plt.yticks()
        plt.ylabel('t-SNE Dimension 1')
        plt.xlabel('t-SNE Dimension 2')

        if savefig:
            plt.savefig('Images_tSNE_Projection.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

    def plot_performance(
        self, 
        metric='acc', 
        combine=False, 
        cv_model=0, 
        ylabel=None, 
        title=None,
        xlim=None, 
        ylim=None, 
        xlog=False, 
        ylog=False, 
        legend_loc=9, 
        savefig=False):
        """
        Plot training (and optional validation) curves for a chosen metric.

        Parameters
        ----------
        metric : {'acc','loss','f1'}, optional
            Metric to visualize: accuracy, loss, or F1 score. Default is 'acc'.
        combine : bool, optional
            If True, include the corresponding validation curves when available.
            Default is False.
        cv_model : int or {'all'}, optional
            Which CV model’s history to plot, or 'all' to overlay every fold.
            Default is 0.
        ylabel : str or None, optional
            Custom y-axis label; if None, derived from `metric`. Default is None.
        title : str or None, optional
            Custom plot title; if None, derived from `metric`. Default is None.
        xlim : tuple or None, optional
            Matplotlib-style (xmin, xmax) limits for the x-axis. Default is None.
        ylim : tuple or None, optional
            Matplotlib-style (ymin, ymax) limits for the y-axis. Default is None.
        xlog : bool, optional
            If True, use a logarithmic x-axis. Default is False.
        ylog : bool, optional
            If True, use a logarithmic y-axis. Default is False.
        legend_loc : int or str, optional
            Legend location (Matplotlib convention). Default is 9.
        savefig : bool, optional
            If True, save to 'CNN_Training_History_<metric>.png' instead of showing.
            Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If histories are not available or `combine=True` without validation metrics.
        """

        if not hasattr(self, 'model_train_metrics'):
            raise ValueError('Training history not found! Run the load() method first!')

        if combine and not hasattr(self, 'model_val_metrics'):
            raise ValueError('combine=True but no validation metrics found!')

        if metric == 'acc':
            index = 0 
        elif metric == 'loss':
            index = 1 
        elif metric == 'f1':
            index = 2
        else:
            raise ValueError('Invalid metric input! Valid options include: "acc", "loss" and "f1"')

        if isinstance(self.model_train_metrics, list) is False or isinstance(cv_model, int):
            metric1 = self.model_train_metrics[cv_model] if isinstance(self.model_train_metrics, list) else self.model_train_metrics
            metric1 = metric1[:,index]
        
            if combine:
                metric2 = self.model_val_metrics[cv_model] if isinstance(self.model_val_metrics, list) else self.model_val_metrics
                metric2 = metric2[:,index]
                label1, label2 = 'Training', 'Validation'
            else:
                label1 = 'Training'
        else: #cv_model='all'
            metric1 = []
            for _metric_ in self.model_train_metrics:
                metric1.append(_metric_[:,index])

            if combine:
                metric2 = []
                for _metric_ in self.model_val_metrics:
                    metric2.append(_metric_[:,index])
                label1, label2 = 'Training', 'Validation'
            else:
                label1 = 'Training'
        
        if cv_model == 'all':
            markers = ['o', 's', '+', 'v', '.', 'x', 'h', 'p', '<', '>', '*']
            color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

            for i in range(len(metric1)):
                marker = markers[i % len(markers)]  # Wrap around the markers list
                plt.plot(range(1, len(metric1[i])+1), metric1[i], color=color[i % len(color)], alpha=0.83, linestyle='-', label=label1+' CV '+str(i+1))

            if combine:
                for i in range(len(metric2)):
                    marker = markers[i % len(markers)]  # Wrap around the markers list
                    plt.plot(range(1, len(metric2[i])+1), metric2[i], color=color[i % len(color)], alpha=0.83, linestyle='--', label=label2+' CV '+str(i+1))
                plt.legend(loc=legend_loc, frameon=False, ncol=2)
            else:
                plt.legend(loc=legend_loc, frameon=False)

        else:
            plt.plot(range(1, len(metric1)+1), metric1, color='r', alpha=0.83, linestyle='-', label=label1)
            if combine:
                plt.plot(range(1, len(metric2)+1), metric2, color='b', alpha=0.83, linestyle='--', label=label2)
                plt.legend(loc=legend_loc, frameon=False, ncol=2)
            else:
                plt.legend(loc=legend_loc, frameon=False)

        if ylabel is None:
            ylabel = metric
        if title is None:
            title = metric

        plt.ylabel(ylabel, alpha=1, color='k')
        plt.title(title)
        plt.xlabel('Epoch', alpha=1, color='k'), plt.grid(False)
        if xlim is not None:
            plt.xlim(xlim)
        else:
            if cv_model == 'all':
                len_ = []
                for _metric_ in self.model_train_metrics:
                    len_.append(len(_metric_))
                plt.xlim(1, np.max(len_))
            else:
                plt.xlim((1, len(metric1)))
        if ylim is not None:
            plt.ylim(ylim)
        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')

        plt.rcParams['axes.facecolor']='white'
        if savefig:
            plt.savefig('CNN_Training_History_'+metric+'.png', bbox_inches='tight', dpi=300)
            plt.clf(); plt.style.use('default')
        else:
            plt.show()

    def _plot_positive(
        self, 
        index=0, 
        channel=0, 
        default_scale=True, 
        vmin=None, 
        vmax=None, 
        cmap='gray', 
        title=''
        ):
        """
        Display a single positive-class image (optionally a single channel or colorized).

        Parameters
        ----------
        index : int, optional
            Index of the sample within the positive class to display. Default is 0.
        channel : int or {'all'}, optional
            Channel index (0-based) to display, or 'all' to show a colorized composite.
            Default is 0.
        default_scale : bool, optional
            If True, use Matplotlib’s default scaling; if False, use `vmin`/`vmax` or
            compute robust limits. Default is True.
        vmin : float or None, optional
            Lower display limit when `default_scale` is False; if None, compute robust
            limits. Default is None.
        vmax : float or None, optional
            Upper display limit when `default_scale` is False; if None, compute robust
            limits. Default is None.
        cmap : str, optional
            Colormap used when displaying a single channel. Default is 'gray'.
        title : str, optional
            Title displayed above the image. Default is ''.

        Returns
        -------
        None
        """

        if len(self.positive_class.shape) == 3:
            data = self.positive_class.reshape(self.positive_class.shape[0], self.positive_class.shape[1], self.positive_class.shape[2], 1)
            data = data[index]
        else:
            data = self.positive_class[index]

        if channel == 'all':
            if vmin is None and default_scale is False:
                plot(data)
            else:
                if default_scale is False:
                    plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
                else:
                    plt.imshow(data[:,:,channel], cmap=cmap); plt.title(title); plt.show()
  
            return 

        if vmin is None and default_scale is False:
            plot(data[:,:,channel])
        else:   
            if default_scale is False:
                plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
            else:
                plt.imshow(data[:,:,channel], cmap=cmap); plt.title(title); plt.show()
          
        return

    def _plot_negative(
        self, 
        index=0, 
        channel=0, 
        default_scale=True, 
        vmin=None, 
        vmax=None, 
        cmap='gray', 
        title=''
        ):
        """
        Display a single negative-class image (optionally a single channel or colorized).

        Parameters
        ----------
        index : int, optional
            Index of the sample within the negative class to display. Default is 0.
        channel : int or {'all'}, optional
            Channel index (0-based) to display, or 'all' to show a colorized composite.
            Default is 0.
        default_scale : bool, optional
            If True, use Matplotlib’s default scaling; if False, use `vmin`/`vmax` or
            compute robust limits. Default is True.
        vmin : float or None, optional
            Lower display limit when `default_scale` is False; if None, compute robust
            limits. Default is None.
        vmax : float or None, optional
            Upper display limit when `default_scale` is False; if None, compute robust
            limits. Default is None.
        cmap : str, optional
            Colormap used when displaying a single channel. Default is 'gray'.
        title : str, optional
            Title displayed above the image. Default is ''.

        Returns
        -------
        None
        """

        if len(self.negative_class.shape) == 3:
            data = self.negative_class.reshape(self.negative_class.shape[0], self.negative_class.shape[1], self.negative_class.shape[2], 1)
            data = data[index]
        else:
            data = self.negative_class[index]

        if channel == 'all':
            if vmin is None and default_scale is False:
                plot(data)
            else:
                if default_scale is False:
                    plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
                else:
                    plt.imshow(data[:,:,channel], cmap=cmap); plt.title(title); plt.show()

            return 

        if vmin is None and default_scale is False:
            plot(data[:,:,channel])
        else:   
            if default_scale is False:
                plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()
            else:
                plt.imshow(data[:,:,channel], vmin=vmin, vmax=vmax, cmap=cmap); plt.title(title); plt.show()

        return

#Custom CNN model configured to generate shallower CNNs than AlexNet

def custom_model(
    positive_class, 
    negative_class, 
    img_num_channels=1, 
    normalize=True, 
    min_pixel=0, 
    max_pixel=100, 
    val_positive=None, 
    val_negative=None, 
    epochs=100, 
    batch_size=32, 
    optimizer='sgd', 
    lr=0.0001, 
    momentum=0.9, 
    decay=0.0, 
    nesterov=False, 
    rho=0.9, 
    beta_1=0.9, 
    beta_2=0.999, 
    amsgrad=False,
    loss='binary_crossentropy', 
    conv_init='uniform_scaling', 
    dense_init='truncated_normal',
    activation_conv='relu', 
    activation_dense='relu', 
    conv_reg=0, 
    dense_reg=0, 
    padding='same', 
    model_reg='batch_norm',
    filter_1=256, 
    filter_size_1=7, 
    strides_1=1, 
    pooling_1='average', 
    pool_size_1=3, 
    pool_stride_1=3, 
    filter_2=0, 
    filter_size_2=0, 
    strides_2=0, 
    pooling_2=None, 
    pool_size_2=0, 
    pool_stride_2=0, 
    filter_3=0, 
    filter_size_3=0, 
    strides_3=0, 
    pooling_3=None, 
    pool_size_3=0, 
    pool_stride_3=0, 
    dense_neurons_1=4096, 
    dropout_1=0.5, 
    dense_neurons_2=0, 
    dropout_2=0, 
    dense_neurons_3=0, 
    dropout_3=0,
    patience=0, 
    metric='binary_accuracy', 
    early_stop_callback=None, 
    checkpoint=False, 
    weight=None, 
    verbose=1, 
    save_training_data=False, 
    path=None
    ):
    """
    Build and train a configurable CNN with 1–3 Conv2D(+pool) blocks followed by up to 3 dense layers.

    The network applies optional normalization, shuffles classes, constructs train/validation
    sets, and trains with optional early stopping and checkpointing. Batch normalization is
    applied after each Conv2D when `model_reg='batch_norm'`.

    Parameters
    ----------
    positive_class : ndarray
        Training images for the positive class. Shape (N, H, W) for single-channel or
        (N, H, W, C) for multi-channel. Required.
    negative_class : ndarray
        Training images for the negative class. Shape (N, H, W) or (N, H, W, C). Required.
    img_num_channels : int, optional
        Number of channels per image (C). Used by preprocessing and input shape.
        Default is 1.
    normalize : bool, optional
        If True, apply min–max normalization using `min_pixel`/`max_pixel`.
        Default is True.
    min_pixel : float, optional
        Lower clip bound used during normalization when `normalize` is True.
        Default is 0.
    max_pixel : float, optional
        Upper clip bound used during normalization when `normalize` is True.
        Default is 100.
    val_positive : ndarray or None, optional
        Validation images for the positive class, same shape convention as training.
        Default is None.
    val_negative : ndarray or None, optional
        Validation images for the negative class, same shape convention as training.
        Default is None.
    epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Mini-batch size. Default is 32.
    optimizer : {'sgd','adam','rmsprop','adadelta','adamw'} or str, optional
        Optimizer name understood by `get_optimizer`. Default is 'sgd'.
    lr : float, optional
        Base learning rate passed to the optimizer. Default is 1e-4.
    momentum : float, optional
        Momentum parameter for SGD-like optimizers. Default is 0.9.
    decay : float, optional
        Learning-rate decay per epoch (if supported by the optimizer). Default is 0.0.
    nesterov : bool, optional
        If True, enable Nesterov momentum for SGD. Default is False.
    rho : float, optional
        Decay factor used by Adadelta/RMSprop-style optimizers. Default is 0.9.
    beta_1 : float, optional
        First-moment decay for Adam-style optimizers. Default is 0.9.
    beta_2 : float, optional
        Second-moment decay for Adam-style optimizers. Default is 0.999.
    amsgrad : bool, optional
        If True, use the AMSGrad variant for Adam-style optimizers. Default is False.
    loss : str, optional
        Loss identifier passed to `get_loss_function` (supports class weighting via `weight`).
        Default is 'binary_crossentropy'.
    conv_init : str or tf.keras.initializers.Initializer, optional
        Convolution kernel initializer. The alias 'uniform_scaling' maps to
        `VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')`.
        Default is 'uniform_scaling'.
    dense_init : str or tf.keras.initializers.Initializer, optional
        Dense kernel initializer. The alias 'uniform_scaling' maps as above.
        Default is 'truncated_normal'.
    activation_conv : str, optional
        Activation applied after each Conv2D (post-BN if enabled). Default is 'relu'.
    activation_dense : str, optional
        Activation applied in dense layers (post-BN if enabled). Default is 'relu'.
    conv_reg : float, optional
        L2 weight for convolution kernels. Default is 0.
    dense_reg : float, optional
        L2 weight for dense kernels. Default is 0.
    padding : {'same','valid'}, optional
        Padding mode for Conv2D and pooling layers. Default is 'same'.
    model_reg : {'batch_norm','local_response',None}, optional
        Per-block regularization: batch normalization or local response normalization
        (LRN after pooling), or None. Default is 'batch_norm'.

    filter_1 : int, optional
        Number of filters in Conv2D block 1. Set ≤0 to disable the block. Default is 256.
    filter_size_1 : int, optional
        Kernel size (square) for block 1 Conv2D. Default is 7.
    strides_1 : int, optional
        Convolution stride for block 1. Default is 1.
    pooling_1 : {'max','average','min',None}, optional
        Pooling type after block-1 Conv2D (custom 'min' pooling supported). Default is 'average'.
    pool_size_1 : int, optional
        Pooling window size for block 1. Default is 3.
    pool_stride_1 : int, optional
        Pooling stride for block 1. Default is 3.

    filter_2 : int, optional
        Number of filters in Conv2D block 2. Set ≤0 to disable the block. Default is 0.
    filter_size_2 : int, optional
        Kernel size (square) for block 2 Conv2D (required if block enabled). Default is 0.
    strides_2 : int, optional
        Convolution stride for block 2 (required if block enabled). Default is 0.
    pooling_2 : {'max','average','min',None}, optional
        Pooling type after block-2 Conv2D. Default is None.
    pool_size_2 : int, optional
        Pooling window size for block 2. Default is 0.
    pool_stride_2 : int, optional
        Pooling stride for block 2. Default is 0.

    filter_3 : int, optional
        Number of filters in Conv2D block 3. Set ≤0 to disable the block. Default is 0.
    filter_size_3 : int, optional
        Kernel size (square) for block 3 Conv2D (required if block enabled). Default is 0.
    strides_3 : int, optional
        Convolution stride for block 3 (required if block enabled). Default is 0.
    pooling_3 : {'max','average','min',None}, optional
        Pooling type after block-3 Conv2D. Default is None.
    pool_size_3 : int, optional
        Pooling window size for block 3. Default is 0.
    pool_stride_3 : int, optional
        Pooling stride for block 3. Default is 0.

    dense_neurons_1 : int, optional
        Units in the first fully-connected (dense) layer. Default is 4096.
    dropout_1 : float, optional
        Dropout rate after the first dense layer (0–1). Default is 0.5.
    dense_neurons_2 : int, optional
        Units in the second dense layer; set ≤0 to skip the layer. Default is 0.
    dropout_2 : float, optional
        Dropout rate after the second dense layer. Default is 0.
    dense_neurons_3 : int, optional
        Units in the third dense layer; set ≤0 to skip the layer. Default is 0.
    dropout_3 : float, optional
        Dropout rate after the third dense layer. Default is 0.

    patience : int, optional
        Early-stopping patience (epochs without improvement on `metric`). A value
        of 0 disables early stopping. Default is 0.
    metric : {'loss','val_loss','binary_accuracy','val_binary_accuracy','f1_score','val_f1_score'}, optional
        Metric monitored by early stopping and checkpointing. The token 'all' is
        coerced internally to 'loss' or 'val_loss'. Default is 'binary_accuracy'.
    early_stop_callback : keras.callbacks.Callback or None, optional
        Additional callback (e.g., from an external optimizer) to signal pruning.
        Default is None.
    checkpoint : bool, optional
        If True, save the best model weights to '~/checkpoint.hdf5' monitored by `metric`.
        Default is False.
    weight : float or None, optional
        Class weight used by certain custom loss wrappers (see `get_loss_function`).
        Default is None.
    verbose : {0,1,2}, optional
        Keras verbosity level (0=silent, 1=progress bar, 2=one line per epoch).
        Default is 1.
    save_training_data : bool, optional
        If True, save the processed training/validation arrays to `path`. Default is False.
    path : str or None, optional
        Directory used when saving training data. If None, the home directory is used.
        Default is None.

    Returns
    -------
    model : tf.keras.Model
        The compiled and trained Keras model (binary sigmoid output).
    history : tf.keras.callbacks.History
        Keras history object containing per-epoch metrics.

    Raises
    ------
    ValueError
        If any enabled convolutional block is missing its required `filter_size_*`
        or `strides_*` arguments.

    Notes
    -----
    - Inputs are shuffled within each class before constructing the training set.
    - When validation data are provided, the same normalization/clipping is applied.
    - Batch normalization can be unstable with very small `batch_size`; if training
      diverges (NaNs), try a larger batch or smaller learning rate.
    - `model_reg='local_response'` inserts LRN after pooling in each enabled block.
    """
    
    if batch_size < 16:
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, 
            min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
        if normalize:
            val_X[val_X > 1] = 1; val_X[val_X < 0] = 0
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2
        if normalize:#FIX
            val_X[val_X > 1] = 1; val_X[val_X < 0] = 0

    img_width, img_height = positive_class[0].shape[0], positive_class[0].shape[1]
     
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

    if Y_train.ndim == 2 and Y_train.shape[1] == 2:
        Y_train = Y_train[:, 1]
        Y_train = Y_train.reshape(-1, 1)

    if val_Y is not None and val_Y.ndim == 2 and val_Y.shape[1] == 2:
        val_Y = val_Y[:, 1]
        val_Y = val_Y.reshape(-1, 1)

    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0
        
    num_classes, input_shape = 1, (img_width, img_height, img_num_channels)

    if verbose != 0:
        filter_size_4 = filter_size_5 = filter_4 = filter_5 = 0; pooling_4 = pool_size_4 = pooling_5 = pool_size_5 ='None' 
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filter_1, filter_2, filter_3, filter_4, filter_5, filter_size_1, filter_size_2, 
            filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pooling_4, pooling_5, pool_size_1, 
            pool_size_2, pool_size_3, pool_size_4, pool_size_5, conv_reg, dense_reg, dense_neurons_1, dense_neurons_2, 
            dense_neurons_3, dropout_1, dropout_2, dropout_3, beta_1, beta_2, amsgrad, rho)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)

    #Model configuration
    model = Sequential()
    
    #Convolutional layers
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=None, input_shape=input_shape, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    #Regularizer: batch_norm
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_conv))
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_1, pool_size_1), strides=(pool_stride_1, pool_stride_1), padding='SAME'))) if pooling_1 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'average' else None
    #Regularizer: local_response, following the AlexNet convention of placing after the pooling layer
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None
    
    if filter_2 > 0:
        if filter_size_2 is None or strides_2 is None:
            raise ValueError('Filter 2 parameters are missing, input the missing arguments.')
        model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        #Regularizer: batch_norm
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
        model.add(Activation(activation_conv))
        #The Pooling Layer
        model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_2, pool_size_2), strides=(pool_stride_2, pool_stride_2), padding='SAME'))) if pooling_2 == 'min' else None
        model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'max' else None
        model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'average' else None
        #Regularizer: local_response, following the AlexNet convention of placing after the pooling layer
        model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None
    
    if filter_3 > 0:
        if filter_size_3 is None or strides_3 is None:
            raise ValueError('Filter 3 parameters are missing, input the missing arguments.')
        model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
        #Regularizer: batch_norm
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
        model.add(Activation(activation_conv))
        #The Pooling Layer
        model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_3, pool_size_3), strides=(pool_stride_3, pool_stride_3), padding='SAME'))) if pooling_3 == 'min' else None
        model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'max' else None
        model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'average' else None
        #Regularizer: local_response, following the AlexNet convention of placing after the pooling layer
        model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None
    
    #Dense layers
    model.add(Flatten())

    #FCC 1
    model.add(Dense(dense_neurons_1, activation=None, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_dense))
    model.add(Dropout(dropout_1))
    
    #FCC 2
    if dense_neurons_2 > 0:
        model.add(Dense(dense_neurons_2, activation=None, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
        model.add(Activation(activation_dense))
        model.add(Dropout(dropout_2))

    #FCC 3
    if dense_neurons_3 > 0:
        model.add(Dense(dense_neurons_3, activation=None, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
        model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
        model.add(Activation(activation_dense))
        model.add(Dropout(dropout_3))

    #Output layer
    model.add(Dense(1, activation='sigmoid', kernel_initializer=dense_init)) 

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])
    
    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train[ix1]); np.save(path+'class_2.npy', X_train[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the pyBIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def AlexNet(
    positive_class, 
    negative_class, 
    img_num_channels=1, 
    normalize=True, 
    min_pixel=0, 
    max_pixel=100, 
    val_positive=None, 
    val_negative=None, 
    epochs=100, 
    batch_size=32, 
    optimizer='sgd', 
    lr=0.0001, 
    momentum=0.9, 
    decay=0.0, 
    nesterov=False, 
    rho=0.9, 
    beta_1=0.9, 
    beta_2=0.999, 
    amsgrad=False,
    loss='binary_crossentropy', 
    conv_init='uniform_scaling', 
    dense_init='truncated_normal',
    activation_conv='relu', 
    activation_dense='relu', 
    conv_reg=0, 
    dense_reg=0, 
    padding='same',
    model_reg='local_response',
    filter_1=96, 
    filter_size_1=11, 
    strides_1=4, 
    pooling_1='max', 
    pool_size_1=3, 
    pool_stride_1=2, 
    filter_2=256, 
    filter_size_2=5, 
    strides_2=1, 
    pooling_2='max', 
    pool_size_2=3, 
    pool_stride_2=2,
    filter_3=384, 
    filter_size_3=3, 
    strides_3=1, 
    pooling_3='max', 
    pool_size_3=3, 
    pool_stride_3=2, 
    filter_4=384, 
    filter_size_4=3, 
    strides_4=1, 
    filter_5=256, 
    filter_size_5=3, 
    strides_5=1, 
    dense_neurons_1=4096, 
    dense_neurons_2=4096, 
    dropout_1=0.5, 
    dropout_2=0.5,  
    patience=0, 
    metric='binary_accuracy', 
    early_stop_callback=None, 
    checkpoint=False, 
    weight=None, 
    verbose=1, 
    save_training_data=False, 
    path=None
    ):
    """
    Build and train an AlexNet-style CNN adapted for binary classification of astronomical images.

    The architecture follows the classic Conv–Pool blocks with Local Response Normalization (LRN) by
    default and provides options for Batch Normalization instead. Inputs can be min–max normalized to
    mitigate gradient issues; the model trains with optional early stopping and checkpointing.

    Parameters
    ----------
    positive_class : ndarray
        Training images for the positive class. Shape (N, H, W) for single-channel or (N, H, W, C) for
        multi-channel. Required.
    negative_class : ndarray
        Training images for the negative class. Shape (N, H, W) or (N, H, W, C). Required.
    img_num_channels : int, optional
        Number of channels per image (C). Used for preprocessing and model input shape. Default is 1.
    normalize : bool, optional
        If True, apply min–max normalization using `min_pixel` and `max_pixel`. Default is True.
    min_pixel : float, optional
        Lower clip bound used during normalization when `normalize` is True. Default is 0.
    max_pixel : float, optional
        Upper clip bound used during normalization when `normalize` is True. Default is 100.
    val_positive : ndarray or None, optional
        Validation images for the positive class, same shape convention as training. Default is None.
    val_negative : ndarray or None, optional
        Validation images for the negative class, same shape convention as training. Default is None.
    epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Mini-batch size. Default is 32.
    optimizer : {'sgd','adam','rmsprop','adadelta','adamw'} or str, optional
        Optimizer name understood by `get_optimizer`. Default is 'sgd'.
    lr : float, optional
        Base learning rate passed to the optimizer. Default is 1e-4.
    momentum : float, optional
        Momentum parameter for SGD-like optimizers. Default is 0.9.
    decay : float, optional
        Learning-rate decay per epoch (if supported by the optimizer). Default is 0.0.
    nesterov : bool, optional
        If True, enable Nesterov momentum for SGD. Default is False.
    rho : float, optional
        Decay factor used by Adadelta/RMSprop-style optimizers. Default is 0.9.
    beta_1 : float, optional
        First-moment decay for Adam-style optimizers. Default is 0.9.
    beta_2 : float, optional
        Second-moment decay for Adam-style optimizers. Default is 0.999.
    amsgrad : bool, optional
        If True, use the AMSGrad variant for Adam-style optimizers. Default is False.
    loss : str, optional
        Loss identifier passed to `get_loss_function` (supports class weighting via `weight`).
        Default is 'binary_crossentropy'.
    conv_init : str or tf.keras.initializers.Initializer, optional
        Convolution kernel initializer. The alias 'uniform_scaling' maps to
        `VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform')`. Default is 'uniform_scaling'.
    dense_init : str or tf.keras.initializers.Initializer, optional
        Dense kernel initializer. The alias 'uniform_scaling' maps as above when used. Default is 'truncated_normal'.
    activation_conv : str, optional
        Activation applied after each Conv2D (post-BN if enabled). Default is 'relu'.
    activation_dense : str, optional
        Activation applied in dense layers (post-BN if enabled). Default is 'relu'.
    conv_reg : float, optional
        L2 weight for convolution kernels. Default is 0.
    dense_reg : float, optional
        L2 weight for dense kernels. Default is 0.
    padding : {'same','valid'}, optional
        Padding mode for Conv2D and pooling layers. Default is 'same'.
    model_reg : {'batch_norm','local_response',None}, optional
        Block-level regularization: Batch Normalization (after Conv2D), Local Response Normalization
        (after pooling, AlexNet-style), or None. Default is 'local_response'.

    filter_1 : int, optional
        Number of filters in Conv2D block 1. Default is 96.
    filter_size_1 : int, optional
        Kernel size (square) for block-1 Conv2D. Default is 11.
    strides_1 : int, optional
        Convolution stride for block-1. Default is 4.
    pooling_1 : {'max','average','min',None}, optional
        Pooling type after block-1 Conv2D (custom 'min' pooling supported). Default is 'max'.
    pool_size_1 : int, optional
        Pooling window size for block-1. Default is 3.
    pool_stride_1 : int, optional
        Pooling stride for block-1. Default is 2.

    filter_2 : int, optional
        Number of filters in Conv2D block 2. Default is 256.
    filter_size_2 : int, optional
        Kernel size (square) for block-2 Conv2D. Default is 5.
    strides_2 : int, optional
        Convolution stride for block-2. Default is 1.
    pooling_2 : {'max','average','min',None}, optional
        Pooling type after block-2 Conv2D. Default is 'max'.
    pool_size_2 : int, optional
        Pooling window size for block-2. Default is 3.
    pool_stride_2 : int, optional
        Pooling stride for block-2. Default is 2.

    filter_3 : int, optional
        Number of filters in Conv2D block 3. Default is 384.
    filter_size_3 : int, optional
        Kernel size (square) for block-3 Conv2D. Default is 3.
    strides_3 : int, optional
        Convolution stride for block-3. Default is 1.
    pooling_3 : {'max','average','min',None}, optional
        Pooling type after block-3 Conv2D (applied after block-5 in this variant). Default is 'max'.
    pool_size_3 : int, optional
        Pooling window size for the final pooling stage. Default is 3.
    pool_stride_3 : int, optional
        Pooling stride for the final pooling stage. Default is 2.

    filter_4 : int, optional
        Number of filters in Conv2D block 4. Default is 384.
    filter_size_4 : int, optional
        Kernel size (square) for block-4 Conv2D. Default is 3.
    strides_4 : int, optional
        Convolution stride for block-4. Default is 1.
    filter_5 : int, optional
        Number of filters in Conv2D block 5. Default is 256.
    filter_size_5 : int, optional
        Kernel size (square) for block-5 Conv2D. Default is 3.
    strides_5 : int, optional
        Convolution stride for block-5. Default is 1.

    dense_neurons_1 : int, optional
        Units in the first fully-connected (dense) layer. Default is 4096.
    dense_neurons_2 : int, optional
        Units in the second fully-connected (dense) layer. Default is 4096.
    dropout_1 : float, optional
        Dropout rate after the first dense layer (0–1). Default is 0.5.
    dropout_2 : float, optional
        Dropout rate after the second dense layer (0–1). Default is 0.5.

    patience : int, optional
        Early-stopping patience (epochs without improvement on `metric`). A value of 0 disables early stopping.
        Default is 0.
    metric : {'loss','val_loss','binary_accuracy','val_binary_accuracy','f1_score','val_f1_score','all'}, optional
        Metric monitored by early stopping and checkpointing. The token 'all' is coerced internally to a single
        monitor ('loss' or 'val_loss'). Default is 'binary_accuracy'.
    early_stop_callback : keras.callbacks.Callback or None, optional
        Additional callback (e.g., from an external optimizer) to signal pruning. Default is None.
    checkpoint : bool, optional
        If True, save the best model weights to '~/checkpoint.hdf5' monitored by `metric`. Default is False.
    weight : float or None, optional
        Class weight used by certain custom loss wrappers (see `get_loss_function`). Default is None.
    verbose : {0,1,2}, optional
        Keras verbosity level (0=silent, 1=progress bar, 2=one line per epoch). Default is 1.
    save_training_data : bool, optional
        If True, save the processed training/validation arrays to `path`. Default is False.
    path : str or None, optional
        Directory used when saving training data. If None, the home directory is used. Default is None.

    Returns
    -------
    model : tf.keras.Model
        The compiled and trained Keras model (single-neuron sigmoid output for binary classification).
    history : tf.keras.callbacks.History
        Keras history object containing per-epoch metrics.

    Notes
    -----
    - Inputs are shuffled within each class before constructing the training set.
    - When validation data are provided, the same normalization/clipping is applied.
    - Batch Normalization can be unstable with very small `batch_size`; if training diverges (NaNs),
      try a larger batch size or a smaller learning rate.
    - `model_reg='local_response'` inserts LRN after pooling in accordance with the original AlexNet paper;
      `model_reg='batch_norm'` places BatchNorm after Conv2D and before activation.
    """
    
    #SEED_NO = 1909

    ##https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development##
    #np.random.seed(SEED_NO), python_random.seed(SEED_NO), tf.random.set_seed(SEED_NO)

    if batch_size < 16 and model_reg == 'batch_norm':
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2

    img_width, img_height = positive_class[0].shape[0], positive_class[0].shape[1]
    
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    if Y_train.ndim == 2 and Y_train.shape[1] == 2:
        Y_train = Y_train[:, 1]
        Y_train = Y_train.reshape(-1, 1)

    if val_Y is not None and val_Y.ndim == 2 and val_Y.shape[1] == 2:
        val_Y = val_Y[:, 1]
        val_Y = val_Y.reshape(-1, 1)

    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0

    num_classes, input_shape = 1, (img_width, img_height, img_num_channels)
    
    if verbose != 0:
        dense_neurons_3 = dropout_3 = 0; pooling_4 = pool_size_4 = pooling_5 = pool_size_5 = 'None' 
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filter_1, filter_2, filter_3, filter_4, filter_5, filter_size_1, filter_size_2, 
            filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pooling_4, pooling_5, pool_size_1, 
            pool_size_2, pool_size_3, pool_size_4, pool_size_5, conv_reg, dense_reg, dense_neurons_1, dense_neurons_2, 
            dense_neurons_3, dropout_1, dropout_2, dropout_3, beta_1, beta_2, amsgrad, rho)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)

    #Model configuration
    model = Sequential()

    #Convolutional layers
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=None, input_shape=input_shape, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    #Regularizer: batch_norm, local_response, or None
    #Convolutional block with batch normalization is set before activation
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_conv))
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_1, pool_size_1), strides=(pool_stride_1, pool_stride_1), padding='SAME'))) if pooling_1 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'average' else None
    #Regularizer: local_response, placed here in accordance with the original AlexNet architecture, in practice batch_norm is placed after conv2d
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    #Regularizer: batch_norm, local_response, or None
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_conv))
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_2, pool_size_2), strides=(pool_stride_2, pool_stride_2), padding='SAME'))) if pooling_2 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'average' else None
    #Regularizer: local_response, placed here in accordance with the original AlexNet architecture, in practice batch_norm is placed after conv2d
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_conv))

    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_conv))

    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_conv))
    #The Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_3, pool_size_3), strides=(pool_stride_3, pool_stride_3), padding='SAME'))) if pooling_3 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'average' else None
    
    #Dense layers
    model.add(Flatten())

    #FCC 1
    model.add(Dense(dense_neurons_1, activation=None, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_dense))
    model.add(Dropout(dropout_1))

    #FCC 2
    model.add(Dense(dense_neurons_2, activation=None, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_dense))
    model.add(Dropout(dropout_2))

    #Output layer (choosing 1 neuron in contrast to 2 neurons for binary classifiers since fewer parameters and computation are needed.)
    model.add(Dense(1, activation='sigmoid', kernel_initializer=dense_init)) #adding a kernel model_reg has no effect if sigmoid is being used, but works for layers with trainable weights like softmax

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])
    
    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train[ix1]); np.save(path+'class_2.npy', X_train[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the pyBIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def VGG16(
    positive_class, 
    negative_class, 
    img_num_channels=1, 
    normalize=True, 
    min_pixel=0, 
    max_pixel=100, 
    val_positive=None, 
    val_negative=None, 
    epochs=100, 
    batch_size=32, 
    optimizer='sgd', 
    lr=0.0001, 
    momentum=0.9, 
    decay=0.0, 
    nesterov=False, 
    rho=0.9, 
    beta_1=0.9, 
    beta_2=0.999, 
    amsgrad=False,
    loss='binary_crossentropy', 
    conv_init='uniform_scaling', 
    dense_init='truncated_normal',
    activation_conv='relu', 
    activation_dense='relu', 
    conv_reg=0, 
    dense_reg=0, 
    padding='same',
    model_reg=None,
    filter_1=64, 
    filter_size_1=3, 
    strides_1=1, 
    pooling_1='max', 
    pool_size_1=2, 
    pool_stride_1=2,
    filter_2=128, 
    filter_size_2=3, 
    strides_2=1,
    pooling_2='max',
    pool_size_2=2, 
    pool_stride_2=2,
    filter_3=256, 
    filter_size_3=3, 
    strides_3=1, 
    pooling_3='max', 
    pool_size_3=2, 
    pool_stride_3=2,
    filter_4=512, 
    filter_size_4=3, 
    strides_4=1, 
    pooling_4='max', 
    pool_size_4=2, 
    pool_stride_4=2,
    filter_5=512, 
    filter_size_5=3, 
    strides_5=1, 
    pooling_5='max', 
    pool_size_5=2, 
    pool_stride_5=2,
    dense_neurons_1=4096, 
    dense_neurons_2=4096, 
    dropout_1=0.5, 
    dropout_2=0.5,
    patience=0, 
    metric='binary_accuracy', 
    early_stop_callback=None, 
    checkpoint=False, 
    weight=None, 
    verbose=1, 
    save_training_data=False, 
    path=None
    ):
    """
    Build and train a VGG16-style CNN for binary classification of astronomical images.

    The model follows five convolutional blocks (with small 3×3 kernels) and two fully-connected
    layers, with configurable pooling and optional Batch Normalization or Local Response
    Normalization after pooling. Inputs can be min–max normalized prior to training.

    Parameters
    ----------
    positive_class : ndarray
        Training images for the positive class. Shape (N, H, W) or (N, H, W, C). Required.
    negative_class : ndarray
        Training images for the negative class. Shape (N, H, W) or (N, H, W, C). Required.
    img_num_channels : int, optional
        Number of channels per image (C) for preprocessing/input shape. Default is 1.
    normalize : bool, optional
        If True, apply min–max normalization using `min_pixel`/`max_pixel`. Default is True.
    min_pixel : float, optional
        Lower clip bound used during normalization when `normalize` is True. Default is 0.
    max_pixel : float, optional
        Upper clip bound used during normalization when `normalize` is True. Default is 100.
    val_positive : ndarray or None, optional
        Validation images for the positive class, same shape convention as training. Default is None.
    val_negative : ndarray or None, optional
        Validation images for the negative class, same shape convention as training. Default is None.
    epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Mini-batch size. Default is 32.
    optimizer : {'sgd','adam','rmsprop','adadelta','adamw'} or str, optional
        Optimizer identifier understood by `get_optimizer`. Default is 'sgd'.
    lr : float, optional
        Base learning rate passed to the optimizer. Default is 1e-4.
    momentum : float, optional
        Momentum for SGD-like optimizers. Default is 0.9.
    decay : float, optional
        Learning-rate decay per epoch (if supported by optimizer). Default is 0.0.
    nesterov : bool, optional
        If True, enable Nesterov momentum for SGD. Default is False.
    rho : float, optional
        Decay factor used by Adadelta/RMSprop-style optimizers. Default is 0.9.
    beta_1 : float, optional
        First-moment decay for Adam-style optimizers. Default is 0.9.
    beta_2 : float, optional
        Second-moment decay for Adam-style optimizers. Default is 0.999.
    amsgrad : bool, optional
        If True, use the AMSGrad variant of Adam. Default is False.
    loss : str, optional
        Loss identifier passed to `get_loss_function` (supports class weighting via `weight`).
        Default is 'binary_crossentropy'.
    conv_init : str or tf.keras.initializers.Initializer, optional
        Convolution kernel initializer; 'uniform_scaling' maps to `VarianceScaling(...)`. Default is 'uniform_scaling'.
    dense_init : str or tf.keras.initializers.Initializer, optional
        Dense kernel initializer; 'uniform_scaling' maps to `VarianceScaling(...)`. Default is 'truncated_normal'.
    activation_conv : str, optional
        Activation applied after each Conv2D (post-BN if enabled). Default is 'relu'.
    activation_dense : str, optional
        Activation applied in dense layers (post-BN if enabled). Default is 'relu'.
    conv_reg : float, optional
        L2 weight for convolution kernels. Default is 0.
    dense_reg : float, optional
        L2 weight for dense kernels. Default is 0.
    padding : {'same','valid'}, optional
        Padding mode for Conv2D and pooling layers. Default is 'same'.
    model_reg : {'batch_norm','local_response',None}, optional
        Block regularization: BatchNorm (after Conv2D), LRN (after pooling), or None. Default is None.

    filter_1 : int, optional
        Number of filters in block-1 Conv2D layers. Default is 64.
    filter_size_1 : int, optional
        Kernel size (square) for block-1 Conv2D. Default is 3.
    strides_1 : int, optional
        Convolution stride for block-1. Default is 1.
    pooling_1 : {'max','average','min',None}, optional
        Pooling type after block-1. Default is 'max'.
    pool_size_1 : int, optional
        Pooling window size for block-1. Default is 2.
    pool_stride_1 : int, optional
        Pooling stride for block-1. Default is 2.

    filter_2 : int, optional
        Number of filters in block-2 Conv2D layers. Default is 128.
    filter_size_2 : int, optional
        Kernel size (square) for block-2 Conv2D. Default is 3.
    strides_2 : int, optional
        Convolution stride for block-2. Default is 1.
    pooling_2 : {'max','average','min',None}, optional
        Pooling type after block-2. Default is 'max'.
    pool_size_2 : int, optional
        Pooling window size for block-2. Default is 2.
    pool_stride_2 : int, optional
        Pooling stride for block-2. Default is 2.

    filter_3 : int, optional
        Number of filters in block-3 Conv2D layers. Default is 256.
    filter_size_3 : int, optional
        Kernel size (square) for block-3 Conv2D. Default is 3.
    strides_3 : int, optional
        Convolution stride for block-3. Default is 1.
    pooling_3 : {'max','average','min',None}, optional
        Pooling type after block-3. Default is 'max'.
    pool_size_3 : int, optional
        Pooling window size for block-3. Default is 2.
    pool_stride_3 : int, optional
        Pooling stride for block-3. Default is 2.

    filter_4 : int, optional
        Number of filters in block-4 Conv2D layers. Default is 512.
    filter_size_4 : int, optional
        Kernel size (square) for block-4 Conv2D. Default is 3.
    strides_4 : int, optional
        Convolution stride for block-4. Default is 1.
    pooling_4 : {'max','average','min',None}, optional
        Pooling type after block-4. Default is 'max'.
    pool_size_4 : int, optional
        Pooling window size for block-4. Default is 2.
    pool_stride_4 : int, optional
        Pooling stride for block-4. Default is 2.

    filter_5 : int, optional
        Number of filters in block-5 Conv2D layers. Default is 512.
    filter_size_5 : int, optional
        Kernel size (square) for block-5 Conv2D. Default is 3.
    strides_5 : int, optional
        Convolution stride for block-5. Default is 1.
    pooling_5 : {'max','average','min',None}, optional
        Pooling type after block-5. Default is 'max'.
    pool_size_5 : int, optional
        Pooling window size for block-5. Default is 2.
    pool_stride_5 : int, optional
        Pooling stride for block-5. Default is 2.

    dense_neurons_1 : int, optional
        Units in the first fully-connected (dense) layer. Default is 4096.
    dense_neurons_2 : int, optional
        Units in the second fully-connected (dense) layer. Default is 4096.
    dropout_1 : float, optional
        Dropout rate after the first dense layer (0–1). Default is 0.5.
    dropout_2 : float, optional
        Dropout rate after the second dense layer (0–1). Default is 0.5.

    patience : int, optional
        Early-stopping patience (epochs without improvement on `metric`); 0 disables early stopping. Default is 0.
    metric : {'loss','val_loss','binary_accuracy','val_binary_accuracy','f1_score','val_f1_score','all'}, optional
        Metric monitored by early stopping/checkpointing; 'all' is coerced internally to a single monitor. Default is 'binary_accuracy'.
    early_stop_callback : keras.callbacks.Callback or None, optional
        Additional callback (e.g., for pruning within external HPO). Default is None.
    checkpoint : bool, optional
        If True, save best model weights to '~/checkpoint.hdf5' monitored by `metric`. Default is False.
    weight : float or None, optional
        Class weight used by certain custom loss wrappers (see `get_loss_function`). Default is None.
    verbose : {0,1,2}, optional
        Keras verbosity level (0=silent, 1=progress bar, 2=one line/epoch). Default is 1.
    save_training_data : bool, optional
        If True, save processed training/validation arrays to `path`. Default is False.
    path : str or None, optional
        Directory used when saving training data; home directory is used if None. Default is None.

    Returns
    -------
    model : tf.keras.Model
        The compiled and trained Keras model (single-neuron sigmoid output for binary classification).
    history : tf.keras.callbacks.History
        Keras history object with per-epoch metrics.

    Notes
    -----
    - Inputs are shuffled within each class prior to constructing the training set.
    - When validation data are provided, the same normalization/clipping is applied.
    - Batch Normalization can be unstable with very small `batch_size`; if training diverges (NaNs),
      try a larger batch size or a smaller learning rate.
    """

    if batch_size < 16 and model_reg == 'batch_norm':
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2

    img_width, img_height = positive_class[0].shape[0], positive_class[0].shape[1]
    
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    if Y_train.ndim == 2 and Y_train.shape[1] == 2:
        Y_train = Y_train[:, 1]
        Y_train = Y_train.reshape(-1, 1)

    if val_Y is not None and val_Y.ndim == 2 and val_Y.shape[1] == 2:
        val_Y = val_Y[:, 1]
        val_Y = val_Y.reshape(-1, 1)

    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0
        
    num_classes, input_shape = 1, (img_width, img_height, img_num_channels)
   
    if verbose != 0:
        dense_neurons_3 = dropout_3 = 'N/A'
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filter_1, filter_2, filter_3, filter_4, filter_5, filter_size_1, filter_size_2, 
            filter_size_3, filter_size_4, filter_size_5, pooling_1, pooling_2, pooling_3, pooling_4, pooling_5, pool_size_1, 
            pool_size_2, pool_size_3, pool_size_4, pool_size_5, conv_reg, dense_reg, dense_neurons_1, dense_neurons_2, 
            dense_neurons_3, dropout_1, dropout_2, dropout_3, beta_1, beta_2, amsgrad, rho)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)

    #Model configuration
    model = Sequential()

    #Block 1
    #Conv2D
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=None, input_shape=input_shape, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None
    model.add(Activation(activation_conv))
    #Conv2D
    model.add(Conv2D(filter_1, filter_size_1, strides=strides_1, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))      
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_1, pool_size_1), strides=(pool_stride_1, pool_stride_1), padding='SAME'))) if pooling_1 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'max' else None 
    model.add(AveragePooling2D(pool_size=pool_size_1, strides=pool_stride_1, padding=padding)) if pooling_1 == 'average' else None 
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 2
    #Conv2D
    model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Conv2D
    model.add(Conv2D(filter_2, filter_size_2, strides=strides_2, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_2, pool_size_2), strides=(pool_stride_2, pool_stride_2), padding='SAME'))) if pooling_2 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_2, strides=pool_stride_2, padding=padding)) if pooling_2 == 'average' else None 
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 3
    #Conv2D
    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Conv2D
    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Conv2D
    model.add(Conv2D(filter_3, filter_size_3, strides=strides_3, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None 
    model.add(Activation(activation_conv)) 
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_3, pool_size_3), strides=(pool_stride_3, pool_stride_3), padding='SAME'))) if pooling_3 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_3, strides=pool_stride_3, padding=padding)) if pooling_3 == 'average' else None
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 4
    #Conv2D
    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Conv2D
    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Conv2D
    model.add(Conv2D(filter_4, filter_size_4, strides=strides_4, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_4, pool_size_4), strides=(pool_stride_4, pool_stride_4), padding='SAME'))) if pooling_4 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_4, strides=pool_stride_4, padding=padding)) if pooling_4 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_4, strides=pool_stride_4, padding=padding)) if pooling_4 == 'average' else None
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Block 5
    #Conv2D
    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None 
    model.add(Activation(activation_conv)) 
    #Conv2D
    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Conv2D
    model.add(Conv2D(filter_5, filter_size_5, strides=strides_5, activation=None, padding=padding, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg)))     
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_conv))
    #Pooling Layer
    model.add(tf.keras.layers.Lambda(lambda x: -tf.nn.max_pool2d(-x, ksize=(pool_size_5, pool_size_5), strides=(pool_stride_5, pool_stride_5), padding='SAME'))) if pooling_5 == 'min' else None
    model.add(MaxPool2D(pool_size=pool_size_5, strides=pool_stride_5, padding=padding)) if pooling_5 == 'max' else None
    model.add(AveragePooling2D(pool_size=pool_size_5, strides=pool_stride_5, padding=padding)) if pooling_5 == 'average' else None
    #Only applying LRN after pooling layers, as per AlexNet convention. 
    model.add(Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75))) if model_reg == 'local_response' else None

    #Dense layers
    model.add(Flatten())

    #FCC 1
    model.add(Dense(dense_neurons_1, activation=None, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_dense))
    model.add(Dropout(dropout_1))
    
    #FCC 2
    model.add(Dense(dense_neurons_2, activation=None, kernel_initializer=dense_init, kernel_regularizer=l2(dense_reg)))
    model.add(BatchNormalization()) if model_reg == 'batch_norm' else None  
    model.add(Activation(activation_dense))
    model.add(Dropout(dropout_2))

    #Output layer
    model.add(Dense(1, activation='sigmoid', kernel_initializer=dense_init)) #adding a kernel model_reg has no effect if sigmoid is being used, but works for layers with trainable weights like softmax

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])

    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train[ix1]); np.save(path+'class_2.npy', X_train[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the pyBIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def Resnet18(
    positive_class, 
    negative_class, 
    img_num_channels=1, 
    normalize=True, 
    min_pixel=0, 
    max_pixel=100, 
    val_positive=None, 
    val_negative=None, 
    epochs=100, 
    batch_size=32, 
    optimizer='sgd', 
    lr=0.0001, 
    momentum=0.9, 
    decay=0.0, 
    nesterov=False, 
    rho=0.9, 
    beta_1=0.9, 
    beta_2=0.999, 
    amsgrad=False,
    loss='binary_crossentropy', 
    conv_init='uniform_scaling', 
    dense_init='truncated_normal',
    activation_conv='relu', 
    activation_dense='relu', 
    conv_reg=0, 
    dense_reg=0, 
    padding='same',
    model_reg=None,
    filters=64, 
    filter_size=7, 
    strides=2, 
    pooling='max', 
    pool_size=3, 
    pool_stride=2,
    block_filters_1=64, 
    block_filters_2=128, 
    block_filters_3=256, 
    block_filters_4=512, 
    block_filters_size=3, 
    patience=0, 
    metric='binary_accuracy', 
    early_stop_callback=None, 
    checkpoint=False, 
    weight=None, 
    verbose=1, 
    save_training_data=False, 
    path=None
    ):#use_zero_padding=True, zero_padding=3, final_avg_pool_size=7
    """
    Build and train a ResNet-18–style CNN with configurable stem, residual block widths,
    optimization hyperparameters, and optional Batch Normalization in the stem.

    Parameters
    ----------
    positive_class : ndarray
        Training images for the positive class, shape (N, H, W) or (N, H, W, C). Default is required.
    negative_class : ndarray
        Training images for the negative class, shape (N, H, W) or (N, H, W, C). Default is required.
    img_num_channels : int, optional
        Number of channels per image (C) for preprocessing/input shape. Default is 1.
    normalize : bool, optional
        If True, apply min–max normalization using `min_pixel`/`max_pixel`. Default is True.
    min_pixel : float, optional
        Lower clip bound used during normalization when `normalize` is True. Default is 0.
    max_pixel : float, optional
        Upper clip bound used during normalization when `normalize` is True. Default is 100.
    val_positive : ndarray or None, optional
        Validation images for the positive class, same shape convention as training. Default is None.
    val_negative : ndarray or None, optional
        Validation images for the negative class, same shape convention as training. Default is None.
    epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Mini-batch size. Default is 32.
    optimizer : {'sgd','adam','rmsprop','adadelta','adamw'} or str, optional
        Optimizer identifier understood by `get_optimizer`. Default is 'sgd'.
    lr : float, optional
        Base learning rate passed to the optimizer. Default is 1e-4.
    momentum : float, optional
        Momentum for SGD-like optimizers. Default is 0.9.
    decay : float, optional
        Learning-rate decay per epoch (if supported by optimizer). Default is 0.0.
    nesterov : bool, optional
        If True, enable Nesterov momentum for SGD. Default is False.
    rho : float, optional
        Decay factor used by Adadelta/RMSprop-style optimizers. Default is 0.9.
    beta_1 : float, optional
        First-moment decay for Adam-style optimizers. Default is 0.9.
    beta_2 : float, optional
        Second-moment decay for Adam-style optimizers. Default is 0.999.
    amsgrad : bool, optional
        If True, use the AMSGrad variant of Adam. Default is False.
    loss : str, optional
        Loss identifier passed to `get_loss_function` (supports class weighting via `weight`). Default is 'binary_crossentropy'.
    conv_init : str or tf.keras.initializers.Initializer, optional
        Convolution kernel initializer; 'uniform_scaling' maps to `VarianceScaling(...)`. Default is 'uniform_scaling'.
    dense_init : str or tf.keras.initializers.Initializer, optional
        Dense kernel initializer; 'truncated_normal' or initializer instance. Default is 'truncated_normal'.
    activation_conv : str, optional
        Activation used inside residual blocks and stem (post-BN if enabled). Default is 'relu'.
    activation_dense : str, optional
        Activation for dense layers (not used in standard ResNet-18 head). Default is 'relu'.
    conv_reg : float, optional
        L2 weight for convolution kernels. Default is 0.
    dense_reg : float, optional
        L2 weight for dense kernels (not used by the default single-neuron head). Default is 0.
    padding : {'same','valid'}, optional
        Convolution padding mode within residual blocks. Default is 'same'.
    model_reg : {'batch_norm', None}, optional
        If 'batch_norm', apply Batch Normalization in the stem and blocks; otherwise omit. Default is None.

    filters : int, optional
        Number of filters in the stem 7×7 convolution. Default is 64.
    filter_size : int, optional
        Kernel size (square) of the stem convolution. Default is 7.
    strides : int, optional
        Stride of the stem convolution. Default is 2.
    pooling : {'max','average','min'}, optional
        Pooling type after the stem convolution. Default is 'max'.
    pool_size : int, optional
        Pooling window size for the stem pooling layer. Default is 3.
    pool_stride : int, optional
        Stride for the stem pooling layer. Default is 2.

    block_filters_1 : int, optional
        Filters in stage-1 residual blocks (two blocks, no downsample). Default is 64.
    block_filters_2 : int, optional
        Filters in stage-2 residual blocks (first block downsamples). Default is 128.
    block_filters_3 : int, optional
        Filters in stage-3 residual blocks (first block downsamples). Default is 256.
    block_filters_4 : int, optional
        Filters in stage-4 residual blocks (first block downsamples). Default is 512.
    block_filters_size : int, optional
        Kernel size (square) for all residual block convolutions. Default is 3.

    patience : int, optional
        Early-stopping patience (epochs without improvement on `metric`); 0 disables early stopping. Default is 0.
    metric : {'loss','val_loss','binary_accuracy','val_binary_accuracy','f1_score','val_f1_score','all'}, optional
        Metric monitored by early stopping/checkpointing; 'all' is coerced internally to a single monitor. Default is 'binary_accuracy'.
    early_stop_callback : keras.callbacks.Callback or None, optional
        Additional callback (e.g., for pruning within external HPO). Default is None.
    checkpoint : bool, optional
        If True, save best model weights to '~/checkpoint.hdf5' monitored by `metric`. Default is False.
    weight : float or None, optional
        Class weight used by certain custom loss wrappers (see `get_loss_function`). Default is None.
    verbose : {0,1,2}, optional
        Keras verbosity level (0=silent, 1=progress bar, 2=one line/epoch). Default is 1.
    save_training_data : bool, optional
        If True, save processed training/validation arrays to `path`. Default is False.
    path : str or None, optional
        Directory used when saving training data; home directory is used if None. Default is None.

    Returns
    -------
    model : tf.keras.Model
        The compiled and trained Keras model (global-average-pooling head with sigmoid output).
    history : tf.keras.callbacks.History
        Keras history object with per-epoch metrics.

    Notes
    -----
    - Inputs are per-class shuffled prior to constructing the training set; optional validation
      arrays receive the same preprocessing and clipping when `normalize` is True.
    - The stem applies `ZeroPadding2D(3)` before the 7×7 stride-2 convolution, followed by pooling.
    - Four residual stages are built as (2, 2, 2, 2) basic blocks; stages 2–4 downsample in the first block.
    - Batch Normalization can be unstable with very small `batch_size`; if training diverges (NaNs),
      try a larger batch size or a smaller learning rate.
    """

    if batch_size < 16 and model_reg == 'batch_norm':
        print("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")
    
    if 'all' in metric: #This is an option for optimization purposes but not a valid argument
        if 'val' in metric:
            print("Cannot combine combined metrics for these callbacks, setting metric='val_loss'"); metric = 'val_loss'
        else:
            print("Cannot combine combined metrics for these callbacks, setting metric='loss'"); metric = 'loss'

    if val_positive is not None:
        val_X1, val_Y1 = process_class(val_positive, label=1, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
        if val_negative is None:
            val_X, val_Y = val_X1, val_Y1
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = np.r_[val_X1, val_X2], np.r_[val_Y1, val_Y2]
    else:
        if val_negative is None:
            val_X, val_Y = None, None
        else:
            val_X2, val_Y2 = process_class(val_negative, label=0, img_num_channels=img_num_channels, min_pixel=min_pixel, max_pixel=max_pixel, normalize=normalize)
            val_X, val_Y = val_X2, val_Y2

    img_width = positive_class[0].shape[0]
    img_height = positive_class[0].shape[1]
    
    ix = np.random.permutation(len(positive_class))
    positive_class = positive_class[ix]

    ix = np.random.permutation(len(negative_class))
    negative_class = negative_class[ix]

    X_train, Y_train = create_training_set(positive_class, negative_class, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    if Y_train.ndim == 2 and Y_train.shape[1] == 2:
        Y_train = Y_train[:, 1]
        Y_train = Y_train.reshape(-1, 1)

    if val_Y is not None and val_Y.ndim == 2 and val_Y.shape[1] == 2:
        val_Y = val_Y[:, 1]
        val_Y = val_Y.reshape(-1, 1)

    if normalize:
        X_train[X_train > 1] = 1; X_train[X_train < 0] = 0
        
    num_classes, input_shape = 1, (img_width, img_height, img_num_channels)
   
    if verbose != 0:
        activation_dense = dropout_3 = 'N/A'; dense_reg = 0
        filter_5 = filter_size_5 = pool_size_5 = dropout_1 = dropout_2 = dense_neurons_1 = dense_neurons_2 = dense_neurons_3 = pooling_2 = pooling_3 = pooling_4 = pooling_5 = 'None'
        print_params(batch_size, lr, decay, momentum, nesterov, loss, optimizer, model_reg, conv_init, activation_conv, 
            dense_init, activation_dense, filters, block_filters_1, block_filters_2, block_filters_3, block_filters_4, 
            filter_size, block_filters_size, block_filters_size, block_filters_size, block_filters_size, pooling, pooling_2, 
            pooling_3, pooling_4, pooling_5, pool_size, pool_size, pool_size, pool_size, pool_size, conv_reg, dense_reg, 
            dense_neurons_1, dense_neurons_2, dense_neurons_3, dropout_1, dropout_2, dropout_3)

    #Uniform scaling initializer
    conv_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if conv_init == 'uniform_scaling' else conv_init
    dense_init = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None) if dense_init == 'uniform_scaling' else dense_init

    #Call the appropriate tf.keras.losses.Loss function
    loss = get_loss_function(loss, weight=weight)


    # Model configuration
    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Stem: zero-pad then 7×7 conv stride=2 then BN then ReLU then 3×3 max-pool stride=2
    x = ZeroPadding2D(padding=3)(x)
    x = Conv2D(filters, kernel_size=filter_size, strides=strides, padding='valid', activation=None, kernel_initializer=conv_init, kernel_regularizer=l2(conv_reg))(x)
    if model_reg == 'batch_norm':
        x = BatchNormalization()(x)

    x = Activation(activation_conv)(x)

    if pooling == 'max':
        x = MaxPool2D(pool_size=pool_size, strides=pool_stride, padding='same')(x)
    elif pooling == 'average':
        x = AveragePooling2D(pool_size=pool_size, strides=pool_stride, padding='same')(x)
    elif pooling == 'min':
        x = Lambda(lambda t: -tf.nn.max_pool2d(-t, ksize=(pool_size, pool_size), strides=(pool_stride, pool_stride), padding='SAME'))(x)

    # Stage 1 (2 × [64 - 64], no downsampling)
    x = resnet_block(x, block_filters_1, block_filters_1, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_1, block_filters_1, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    # Stage 2 (2 × [64 - 128], downsample in first)
    x = resnet_block(x, block_filters_1, block_filters_2, block_filters_size, activation=activation_conv, stride=2, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_2, block_filters_2, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    # Stage 3
    x = resnet_block(x, block_filters_2, block_filters_3, block_filters_size, activation=activation_conv, stride=2, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_3, block_filters_3, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    # Stage 4
    x = resnet_block(x, block_filters_3, block_filters_4, block_filters_size, activation=activation_conv, stride=2, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)
    x = resnet_block(x, block_filters_4, block_filters_4, block_filters_size, activation=activation_conv, stride=1, padding=padding, kernel_initializer=conv_init, model_reg=model_reg)

    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid', kernel_initializer=dense_init)(x)

    model = Model(inputs=input_tensor, outputs=x)

    #Call the appropriate tf.keras.optimizers function
    optimizer = get_optimizer(optimizer, lr, momentum=momentum, decay=decay, rho=rho, nesterov=nesterov, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)

    #Compile the Model
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1_score])

    #Wheter to maximize or minimize the metric
    mode = 'min' if 'loss' in metric else 'max'

    #Optional checkpoint callback, with the monitor being the input metric.
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(str(Path.home())+'/'+'checkpoint.hdf5', monitor=metric, verbose=2, save_best_only=True, mode=mode)) if checkpoint else None

    #Early stopping callback
    callbacks_list.append(EarlyStopping(monitor=metric, mode=mode, patience=patience)) if patience > 0 else None

    #Early stop callback for use during the optimization routine
    callbacks_list.append(early_stop_callback) if early_stop_callback is not None else None

    #Fit the Model
    if val_X is None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=verbose)
    else:
        history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=verbose)

    if save_training_data:
        path = str(Path.home()) if path is None else path
        path += '/' if path[-1] != '/' else ''
        try:
            flattened_labels = np.argmax(Y_train, axis=1)
            ix1 = np.where(flattened_labels == 1)[0]; ix2 = np.where(flattened_labels == 0)[0]; 
            np.save(path+'class_1.npy', X_train[ix1]); np.save(path+'class_2.npy', X_train[ix2]); 
            if val_positive is not None:
                np.save(path+'val_class_1.npy', val_X1)
            if val_negative is not None:
                np.save(path+'val_class_2.npy', val_X2)     
            print('Files saved in: {}'.format(path)); print('NOTE: The training data files may have to be manually moved to the pyBIA_cnn_model folder in order for them to be loaded when running load()!') 
        except Exception as e:
            print(f"Could not save training data due to error: {e}")

    return model, history

def resnet_block(
    x, 
    filters_in, 
    filters_out, 
    filter_size=3, 
    activation='relu', 
    stride=1, 
    padding='same', 
    kernel_initializer='he_normal', 
    model_reg='batch_norm'
    ):
    """
    Basic residual block with two conv layers and an identity/projection skip.

    Parameters
    ----------
    x : tf.Tensor
        Input feature map tensor to be transformed by the block. Default is required.
    filters_in : int
        Number of channels in the input tensor (used to decide if projection is needed). Default is required.
    filters_out : int
        Number of output channels for both convolutions (and the projection, if used). Default is required.
    filter_size : int, optional
        Square kernel size for both convolutions in the block. Default is 3.
    activation : str, optional
        Activation applied after BatchNorm (or directly after conv if BN is disabled). Default is 'relu'.
    stride : int, optional
        Stride of the first convolution (and projection); controls downsampling. Default is 1.
    padding : {'same','valid'}, optional
        Convolution padding for both conv layers in the block. Default is 'same'.
    kernel_initializer : str or tf.keras.initializers.Initializer, optional
        Kernel initializer for all conv layers and the projection. Default is 'he_normal'.
    model_reg : {'batch_norm', None}, optional
        If 'batch_norm', apply BatchNormalization after each conv; otherwise omit normalization. Default is 'batch_norm'.

    Returns
    -------
    tf.Tensor
        Output tensor after two convolutions and residual addition (with projection when shape/stride differs).

    Notes
    -----
    - A 1×1 projection on the skip path is used when `stride != 1` or `filters_in != filters_out`.
    - The final activation is applied after the residual addition.
    """

    residual = x

    # First conv layer
    x = Conv2D(filters_out, filter_size, strides=stride, padding=padding,
               activation=None, kernel_initializer=kernel_initializer)(x)
    if model_reg == 'batch_norm':
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # Second conv layer
    x = Conv2D(filters_out, filter_size, strides=1, padding=padding,
               activation=None, kernel_initializer=kernel_initializer)(x)
    if model_reg == 'batch_norm':
        x = BatchNormalization()(x)

    # Shortcut projection if we change shape
    if stride != 1 or filters_in != filters_out:
        residual = Conv2D(filters_out, 1, strides=stride, padding='valid',
                          activation=None, kernel_initializer=kernel_initializer)(residual)
        if model_reg == 'batch_norm':
            residual = BatchNormalization()(residual)

    # Merge & final ReLU
    x = Add()([x, residual])
    x = Activation(activation)(x)

    return x

    
### Score and Loss Functions ###

def f1_score(y_true, y_pred):
    """
    Binary F1 score (harmonic mean of precision and recall) for the current batch.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth binary labels with shape (N,) or (N, 1). Values are expected to be 0 or 1.
    y_pred : tf.Tensor
        Model outputs with shape matching `y_true`. Values are expected to be probabilities in [0, 1]
        (e.g., sigmoid outputs). A fixed threshold of 0.5 is applied internally via rounding.

    Returns
    -------
    tf.Tensor
        Scalar tensor containing the batch F1 score in [0, 1].
    """
    
    tp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred - y_true, 0, 1)))
    fn = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())

    f1 = 2.0 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    return f1

def calculate_tp_fp(model, sample, y_true):
    """
    Compute batch true positives (TP) and false positives (FP) from model predictions.

    The model's predicted probabilities are thresholded at 0.5 (via rounding) to obtain
    binary predictions. Labels are clipped to [0, 1] and rounded. Sums are computed
    over the batch, returning scalar tensors.

    Parameters
    ----------
    model : tf.keras.Model or compatible
        Trained model providing `predict(sample)` → probabilities for the positive class.
    sample : ndarray or tf.Tensor
        Input batch for inference, typically with shape (N, H, W, C) or (N, d).
        No default; must be provided.
    y_true : ndarray or tf.Tensor
        Ground-truth labels for `sample`. Shape (N,) or (N, 1) with values in {0, 1}.
        One-hot labels must be preconverted to a single positive column. No default.

    Returns
    -------
    tp : tf.Tensor
        Scalar tensor equal to the count of true positives in the batch.
    fp : tf.Tensor
        Scalar tensor equal to the count of false positives in the batch.
    """

    # Make a prediction using the model
    y_pred = model.predict(sample)

    # Convert y_true and y_pred to binary values
    y_true_binary = tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1))
    y_pred_binary = tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1))

    # Calculate true positives (tp)
    tp = tf.keras.backend.sum(y_true_binary * y_pred_binary)

    # Calculate false positives (fp)
    fp = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred_binary - y_true_binary, 0, 1)))

    return tp, fp

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Binary focal loss for imbalanced classification (Lin et al., 2017).

    Down-weights easy examples and focuses training on hard, misclassified ones.
    This implementation uses binary cross-entropy with `from_logits=True`, so
    `y_pred` must be raw logits (pre-sigmoid).

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth binary labels in {0, 1}; shape broadcastable to `y_pred`.
        No default; must be provided.
    y_pred : tf.Tensor
        Model outputs **as logits** (before sigmoid); same shape as `y_true`.
        No default; must be provided.
    gamma : float, optional
        Focusing parameter; larger values increase down-weighting of easy
        examples. Default is 2.0.
    alpha : float, optional
        Global weighting factor for the loss (often the positive-class weight).
        Default is 0.25.

    Returns
    -------
    tf.Tensor
        Element-wise focal loss with the same shape as `y_true`. Reduce with
        `tf.reduce_mean` or `tf.reduce_sum` for a scalar loss.
    """

    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)
    pt = tf.math.exp(-ce)

    return alpha * tf.math.pow(1.0 - pt, gamma) * ce

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
    Dice loss for binary segmentation.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth binary mask in {0, 1}; shape broadcastable to `y_pred`. No default; must be provided.
    y_pred : tf.Tensor
        Predicted mask as probabilities in [0, 1] (apply sigmoid if logits); same shape as `y_true`. No default; must be provided.
    smooth : float, optional
        Smoothing constant added to numerator and denominator for numerical stability. Default is 1e-7.

    Returns
    -------
    tf.Tensor
        Scalar Dice loss for the batch (1 − Dice coefficient).
    """

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice

def jaccard_loss(y_true, y_pred, smooth=1e-7):
    """
    Jaccard (IoU) loss for binary segmentation.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth binary mask in {0, 1}; shape broadcastable to `y_pred`. No default; must be provided.
    y_pred : tf.Tensor
        Predicted mask as probabilities in [0, 1] (apply sigmoid if logits); same shape as `y_true`. No default; must be provided.
    smooth : float, optional
        Smoothing constant added to numerator and denominator for numerical stability. Default is 1e-7.

    Returns
    -------
    tf.Tensor
        Scalar Jaccard loss for the batch (1 − IoU).
    """

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard = (intersection + smooth) / (union + smooth)

    return 1.0 - jaccard

def weighted_binary_crossentropy(weight):
    """
    Weighted binary cross-entropy (positive-class scaling).

    Returns a Keras-compatible loss function that computes binary cross-entropy
    with the positive term multiplied by `weight`. Use this to counter class
    imbalance by up-weighting positives (weight > 1) or down-weighting them
    (0 < weight < 1).

    Parameters
    ----------
    weight : float
        Non-negative scalar applied to the positive class term.
        Values > 1 increase the penalty for false negatives; values in (0, 1)
        decrease it. No default; must be provided.

    Returns
    -------
    Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
        A loss function `loss(y_true, y_pred)` that returns the mean weighted
        binary cross-entropy over the last axis.
    """

    def loss(y_true, y_pred):
        """
        Weighted binary cross-entropy per sample (positive class scaled by `weight`).

        Returns
        -------
        tensor
            Mean weighted binary cross-entropy over the last axis for each sample.
        """
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        bce = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
        weighted_bce = K.mean(bce, axis=-1)
        return weighted_bce

    return loss


### AlexNet Helper Functions ###

def get_optimizer(
    optimizer, 
    lr, 
    momentum=None, 
    decay=None, 
    rho=0.9, 
    nesterov=False, 
    beta_1=0.9, 
    beta_2=0.999, 
    amsgrad=False
    ):
    """
    Return a configured Keras optimizer instance.

    Parameters
    ----------
    optimizer : {'sgd','adam','adamax','nadam','adadelta','rmsprop'}
        Optimizer name to instantiate. No default.
    lr : float
        Learning rate for the optimizer. No default.
    momentum : float, optional
        Momentum term for SGD (ignored by other optimizers). Default is None.
    decay : float, optional
        Learning-rate time decay (not used by this function). Default is None.
    rho : float, optional
        Discounting factor for the moving average of squared grads
        (Adadelta/RMSprop). Default is 0.9.
    nesterov : bool, optional
        Use Nesterov momentum with SGD. Default is False.
    beta_1 : float, optional
        Exponential decay rate for first-moment estimates (Adam-family). Default is 0.9.
    beta_2 : float, optional
        Exponential decay rate for second-moment estimates (Adam-family). Default is 0.999.
    amsgrad : bool, optional
        Use the AMSGrad variant of Adam. Default is False.

    Returns
    -------
    optimizer
        A compiled `tf.keras.optimizers.Optimizer` instance.

    Raises
    ------
    ValueError
        If `optimizer` is not one of the supported names.
    """

    if optimizer == 'sgd':
        optimizer = SGD(learning_rate=lr,  momentum=momentum, nesterov=nesterov)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, amsgrad=amsgrad)
    elif optimizer == 'adamax':
        optimizer = Adamax(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'nadam':
        optimizer = Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(learning_rate=lr, rho=rho)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr, rho=rho)
    else:
        raise ValueError("Invalid optimizer name. Available options are 'sgd', 'adam', 'adamax', 'nadam', 'adadelta', or 'rmsprop'.")

    return optimizer

def get_loss_function(loss, weight=None):
    """
    Return a Keras-compatible loss given a symbolic name.

    Parameters
    ----------
    loss : {'binary_crossentropy','hinge','squared_hinge','kld','logcosh','focal_loss','dice_loss','jaccard_loss','weighted_binary_crossentropy'}
        Name of the loss to construct. No default.
    weight : float or None, optional
        Positive-class weight used only when `loss='weighted_binary_crossentropy'`;
        ignored for all other losses. Default is None.

    Returns
    -------
    loss_fn : str or tf.keras.losses.Loss or callable
        Keras-compatible loss object. This may be a string identifier
        (for `'binary_crossentropy'`), a `tf.keras.losses.*` instance
        (e.g., `Hinge()`, `KLDivergence()`, `LogCosh()`), or a callable
        such as `focal_loss`, `dice_loss`, `jaccard_loss`, or the result of
        `weighted_binary_crossentropy(weight)`.
    """

    if loss == 'binary_crossentropy':
        return loss 
    elif loss == 'hinge':
        return Hinge()
    elif loss == 'squared_hinge':
        return SquaredHinge()
    elif loss == 'kld':
        return KLDivergence()
    elif loss == 'logcosh':
        return LogCosh()
    elif loss == 'focal_loss':
        return focal_loss
    elif loss == 'dice_loss':
        return dice_loss
    elif loss == 'jaccard_loss':
        return jaccard_loss
    elif loss == 'weighted_binary_crossentropy':
        if weight is None:
            raise ValueError('If using weighted loss function, the weight parameter must be input!')
        return weighted_binary_crossentropy(weight)
    else:
        raise ValueError("Invalid loss function name")
   
def print_params(
    batch_size, 
    lr, 
    decay, 
    momentum, 
    nesterov, 
    loss, 
    optimizer, 
    model_reg, 
    conv_init, 
    activation_conv, 
    dense_init, 
    activation_dense,
    filter1, 
    filter2, 
    filter3, 
    filter4, 
    filter5, 
    filter_size_1, 
    filter_size_2, 
    filter_size_3, 
    filter_size_4, 
    filter_size_5, 
    pooling_1, 
    pooling_2, 
    pooling_3,
    pooling_4, 
    pooling_5, 
    pool_size_1, 
    pool_size_2, 
    pool_size_3, 
    pool_size_4, 
    pool_size_5,
    conv_reg, 
    dense_reg, 
    dense_neurons_1, 
    dense_neurons_2, 
    dense_neurons_3, 
    dropout_1, 
    dropout_2, 
    dropout_3, 
    beta_1, 
    beta_2, 
    amsgrad, 
    rho
    ):
    """
    Print a formatted summary of training hyperparameters and model architecture settings.

    Parameters
    ----------
    batch_size : int
        Number of samples per gradient update. No default.
    lr : float
        Optimizer learning rate. No default.
    decay : float
        Learning-rate time decay (per update/epoch, depending on optimizer). No default.
    momentum : float
        SGD momentum coefficient. No default.
    nesterov : bool
        Whether SGD uses Nesterov momentum. No default.
    loss : str
        Name of the loss function (e.g., 'binary_crossentropy'). No default.
    optimizer : str
        Optimizer identifier (e.g., 'sgd','adam','rmsprop','adadelta','adamax','nadam'). No default.
    model_reg : str or None
        Model-level regularization flag (e.g., 'batch_norm','local_response', or None). No default.
    conv_init : str or tf.keras.initializers.Initializer
        Convolutional kernel initializer (e.g., 'he_normal','glorot_uniform','uniform_scaling'). No default.
    activation_conv : str
        Activation function used after convolutional layers (e.g., 'relu'). No default.
    dense_init : str or tf.keras.initializers.Initializer
        Dense layer kernel initializer. No default.
    activation_dense : str
        Activation function used in dense layers (e.g., 'relu','tanh'). No default.

    filter1 : int
        Number of filters in convolutional layer/block 1; zero disables the layer. No default.
    filter2 : int
        Number of filters in convolutional layer/block 2; zero disables the layer. No default.
    filter3 : int
        Number of filters in convolutional layer/block 3; zero disables the layer. No default.
    filter4 : int
        Number of filters in convolutional layer/block 4; zero disables the layer. No default.
    filter5 : int
        Number of filters in convolutional layer/block 5; zero disables the layer. No default.

    filter_size_1 : int
        Kernel size for convolutional layer/block 1. No default.
    filter_size_2 : int
        Kernel size for convolutional layer/block 2. No default.
    filter_size_3 : int
        Kernel size for convolutional layer/block 3. No default.
    filter_size_4 : int
        Kernel size for convolutional layer/block 4. No default.
    filter_size_5 : int
        Kernel size for convolutional layer/block 5. No default.

    pooling_1 : {'max','average','min', None}
        Pooling mode after layer/block 1; None disables pooling. No default.
    pooling_2 : {'max','average','min', None}
        Pooling mode after layer/block 2; None disables pooling. No default.
    pooling_3 : {'max','average','min', None}
        Pooling mode after layer/block 3; None disables pooling. No default.
    pooling_4 : {'max','average','min', None}
        Pooling mode after layer/block 4; None disables pooling. No default.
    pooling_5 : {'max','average','min', None}
        Pooling mode after layer/block 5; None disables pooling. No default.

    pool_size_1 : int
        Pool window size after layer/block 1 (if pooling enabled). No default.
    pool_size_2 : int
        Pool window size after layer/block 2 (if pooling enabled). No default.
    pool_size_3 : int
        Pool window size after layer/block 3 (if pooling enabled). No default.
    pool_size_4 : int
        Pool window size after layer/block 4 (if pooling enabled). No default.
    pool_size_5 : int
        Pool window size after layer/block 5 (if pooling enabled). No default.

    conv_reg : float
        L2 regularization coefficient applied to convolutional kernels. No default.
    dense_reg : float
        L2 regularization coefficient applied to dense kernels. No default.

    dense_neurons_1 : int
        Number of units in dense layer 1. No default.
    dense_neurons_2 : int
        Number of units in dense layer 2; zero disables the layer. No default.
    dense_neurons_3 : int
        Number of units in dense layer 3; zero disables the layer. No default.

    dropout_1 : float
        Dropout rate applied after dense layer 1 (0–1). No default.
    dropout_2 : float
        Dropout rate applied after dense layer 2 (0–1). No default.
    dropout_3 : float or str
        Dropout rate after dense layer 3 (0–1); may be 'N/A' for models without this layer. No default.

    beta_1 : float
        Adam/Nadam first-moment decay (β₁). No default.
    beta_2 : float
        Adam/Nadam second-moment decay (β₂). No default.
    amsgrad : bool
        Whether to use the AMSGrad variant of Adam. No default.
    rho : float
        Exponential decay factor for Adadelta/RMSprop. No default.

    Returns
    -------
    None
        This function prints to stdout and returns nothing.
    """

    print(); print('===== Training Parameters ====='); print()
    print('|| Batch Size : '+str(batch_size), '|| Loss Function : '+loss, '||')

    if optimizer == 'sgd':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(np.round(lr, 7)), '|| Decay : '+str(np.round(decay, 5)), '|| Momentum : '+str(momentum), '|| Nesterov : '+str(nesterov)+' ||')
    elif optimizer == 'adadelta' or optimizer == 'rmsprop':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(np.round(lr, 7)), '|| rho : '+str(np.round(rho, 5)), '|| Decay : '+str(np.round(decay, 5))+' ||')
    elif optimizer == 'adam' or optimizer == 'adamax' or optimizer == 'nadam':
        print('|| Optimizer : '+optimizer, '|| lr : '+str(np.round(lr, 7)), '|| Beta 1 : '+str(np.round(beta_1, 5)), '|| Beta 2 : '+str(np.round(beta_2, 5)), '||  amsgrad : '+str(amsgrad)+' ||')
    
    print(); print('=== Architecture Parameters ==='); print()
    print('Model Regularizer : '+ str(model_reg))
    print('Convolutional L2 Regularizer : '+ str(conv_reg))
    print('Convolutional Initializer : '+ conv_init)
    print('Convolutional Activation Fn : '+ activation_conv)
    print('Dense L2 Regularizer : '+ str(dense_reg))
    print('Dense Initializer : '+ dense_init)
    print('Dense Activation Fn : '+ activation_dense); print()

    if dropout_3 != 'N/A': #This is AlexNet and custom_model since droput_3 = N/A is set for VGG16 and Resnet-18 only
        print('======= Conv2D Layer Parameters ======'); print()
        print('Filter 1 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter1, filter_size_1, pooling_1, pool_size_1))
        if filter_size_2 > 0:
            print('Filter 2 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter2, filter_size_2, pooling_2, pool_size_2))
        if filter_size_3 > 0:
            print('Filter 3 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter3, filter_size_3, pooling_3, pool_size_3))
        if filter_size_4 > 0:
            print('Filter 4 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter4, filter_size_4, pooling_4, pool_size_4))
        if filter_size_5 > 0:
            print('Filter 5 || Num: {}, Size : {}, Pooling : {}, Pooling Size : {}'.format(filter5, filter_size_5, pooling_5, pool_size_5))
        
        print(); print('======= Dense Layer Parameters ======'); print()
        print('Neurons 1 || Num : {}, Dropout : {}'.format(dense_neurons_1, dropout_1))
        if dense_neurons_2 > 0:
            print('Neurons 2 || Num : {}, Dropout : {}'.format(dense_neurons_2, dropout_2))
        if dense_neurons_3 > 0:
            print('Neurons 3 || Num : {}, Dropout : {}'.format(dense_neurons_3, dropout_3))
        print(); print('==============================='); print()
    else:
        if activation_dense == 'N/A': #For Resnet-18 
            print('======= Conv2D Layer Parameters ======'); print()
            print('Filter 1 || Num: {}, Size : {}, Pooling : {}'.format(filter1, filter_size_1, pooling_1, pool_size_1))
            if filter_size_2 > 0:
                print('Residual Block 1 || Num: {}, Size : {}'.format(filter2, filter_size_2))
            if filter_size_3 > 0:
                print('Residual Block 2 || Num: {}, Size : {}'.format(filter3, filter_size_3))
            if filter_size_4 > 0:
                print('Residual Block 3 || Num: {}, Size : {}'.format(filter4, filter_size_4))
            if filter_size_5 > 0:
                print('Residual Block 4 || Num: {}, Size : {},'.format(filter5, filter_size_5))
            print(); print('==============================='); print()
        else: #For VGG16
            print('======= Conv2D Layer Parameters ======'); print()
            print('Block 1 || Num: {}, Size : {}, Pooling : {}'.format(filter1, filter_size_1, pooling_1, pool_size_1))
            print('Block 2 || Num: {}, Size : {}'.format(filter2, filter_size_2))
            print('Block 3 || Num: {}, Size : {}'.format(filter3, filter_size_3))
            print('Block 4 || Num: {}, Size : {}'.format(filter4, filter_size_4))
            print('Block 5 || Num: {}, Size : {},'.format(filter5, filter_size_5))
            print(); print('======= Dense Layer Parameters ======'); print()
            print('Neurons 1 || Num : {}, Dropout : {}'.format(dense_neurons_1, dropout_1))
            print('Neurons 2 || Num : {}, Dropout : {}'.format(dense_neurons_2, dropout_2))    
            print(); print('==============================='); print()

def format_labels(labels: list) -> list:
    """
    Convert raw parameter keys into human-readable display labels.

    Parameters
    ----------
    labels : list of str
        Sequence of raw label strings to format.

    Returns
    -------
    list of str
        List of formatted labels (same order and length as input).

    Notes
    -----
    This function applies a set of explicit replacements; if no rule matches,
    the label is converted by replacing underscores with spaces and applying
    title case.
    """
    new_labels = []
    for label in labels:
        if label == "lr":
            new_labels.append("Learning Rate")
            continue
        if label == "max_pixel_1":
            new_labels.append(r"$B_W$ Max Pixel")
            continue
        if label == "max_pixel_2":
            new_labels.append(r"$R$ Max Pixel")
            continue
        if label == "max_pixel_3":
            new_labels.append(r"$B_W \ / \ R$ Max Pixel")
            continue
        if label == "num_aug":
            new_labels.append("No. of Augmentations")
            continue
        if label == "activation_conv":
            new_labels.append("Conv2D Activation")
            continue
        if label == "activation_dense":
            new_labels.append("FC Activation")
            continue
        if label == "loss":
            new_labels.append("Loss Function")
            continue
        if label == "dense_init":
            new_labels.append("FC Init.")
            continue
        if label == "conv_init":
            new_labels.append("Conv2D Init.")
            continue
        if label == "beta_1":
            new_labels.append(r"$\beta_1")
            continue
        if label == "beta_2":
            new_labels.append(r"$\beta_2")
            continue
        if label == "amsgrad":
            new_labels.append("AMSGrad")
        if label == "optimizer":
            new_labels.append("Optimizer")
            continue
        if label == 'model_reg':
            new_labels.append("Regularizer")
            continue 
        label = label.replace("_", " ")
        new_labels.append(label.title())

    return new_labels
