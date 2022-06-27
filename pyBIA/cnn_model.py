#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:40:39 2021

@author: daniel
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from pathlib import Path
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from pyBIA.data_processing import process_class, create_training_set

def bw_model():
    """
    Calling this will load the trained Tensorflow model, trained using NDWFS images
    in the blue broadband.
    
    Note:
        Training new models with 1000 epochs can take over a week, this Bw model
        was trained using NDWFS blue broadband images of the Bootes field. The 
        corresponding .h5 file is located in the data folder inside the pyBIA directory 
        in the Python path. 

    Returns:
        The pyBIA CNN model used for classifying images in blue broadband surveys.

    """
    import tensorflow as tf
    from keras.models import load_model
    import pkg_resources

    resource_package = __name__
    resource_path = '/'.join(('data', 'Bw_CNN_Model.h5'))
    model = tf.keras.models.load_model(pkg_resources.resource_filename(resource_package, resource_path))
    print('Bw model successfully loaded.')
    print('Note: Input data when using this model must be 50x50.')
    return model

def create(blob_data, other_data, img_num_channels=1, normalize=True, 
    min_pixel=638, max_pixel=3000, val_X=None, val_Y=None, 
    epochs=100, batch_size=32, lr=0.0001, batch_norm=True, momentum=0.9, decay=0.0005, 
    nesterov=False, loss='categorical_crossentropy', padding='same', 
    dropout=0.5, pooling=True, metrics=True, filename=''):
    """
    The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
    Visual Recognition Challenge, AlexNet. Parameters were adapted for
    our astronomy case of detecting diffuse objects.

    Note:
        To avoid exploding gradients we need to normalize our pixels to be 
        between 0 and 1. By default normalize=True, which will perform
        min-max normalization using the min_pixel and max_pixel arguments, 
        which should be set carefully.

        The min_pixel parameter is set to 0 by default as the data is assumed
        to be background-subtracted. The max_pixel must be adequately brighter
        than the brighest expected target object. In this example we expected
        the high redshift Lyman-alpha nebulae to appear diffuse and less bright,
        so anything brighter than max_pixel=2000 can be categorized as too bright 
        to be a candidate source.
    
    Args:
        blob_data (ndarray): 3D array containing more than one image of diffuse objects.
        other_data (ndarray): 3D array containing more than one image of non-diffuse objects.
        img_num_channels (int): The number of filters used. Defaults to 1, as pyBIA version 1
            has been trained with only blue broadband data.
        normalize (bool, optional): True will normalize the data using the input min and max pixels
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.
        val_X (array, optional): 3D matrix containing the 2D arrays (images)
            to be used for validation.
        val_Y (array, optional): A binary class matrix containing the labels of the
            corresponding validation data. This binary matrix representation can be created
            using tensorflow, see example in the Notes.
        epochs (int): Number of epochs used for training. 
        batch_size (int): The size of each sub-sample used during the training
            epoch. Large batches are likely to get stuck in local minima. Defaults to 32.
        lr (float): Learning rate, the rate at which the model updates the gradient. Defaults to 0.0001
        batch_norm (bool): If False, batch normalization after each max pooling layer
            is disabled. Defaults to True.
            Note: If set to True it might be best also set dropout=0. See: https://arxiv.org/abs/1502.03167
        momentum (float): Momentum is a float greater than 0 that accelerates gradient descent. Defaults to 0.9.
        decay (float): The rate of learning rate decay applied after each epoch. Defaults to 0.0005.
        nesterov (bool): Whether to apply Nesterov momentum or not. Defaults to False.
        loss (str): The loss function used to calculate the gradients. Defaults to 'categorical_crossentropy'.
            Loss functions can be set by calling the Keras API losses module.
        padding (str): Either 'same' or 'valid'. When set to 'valid', the dimensions reduce as the boundary 
            that doesn't make it within even convolutions get cuts off. Defaults to 'same', which applies
            zero-value padding around the boundary, ensuring even convolutional steps across each dimension.
        dropout (float): Droupout rate between 0 and 1. This is the percentage of dense neurons
            that are turned off at each epoch. This prevents inter-neuron depedency, and thus overfitting. 
            Note: If batch_norm=True, it might be best to disable droput. See: https://arxiv.org/abs/1502.03167
        pooling (bool): True to enable max pooling, false to disable. 
            Note: Max pooling can result in loss of positional information, it computation allows
            setting pooling=False may yield more robust accuracy.
        metrics (bool): When False the CNN will not save the model accuracy and loss. Defaults to True.
        filename (str, optional): The name of the metrics filename. The metrics files will be saved
            as 'model_metric'+filename.

    Example:
        To use a validation dataset when training the model, the val_X and val_Y
        parameters must be input. The val_X is a 3D matrix containing all the images, and
        the val_Y is another matrix containing their class label (0 for DIFFUSE, 1 for OTHER).

        If you have validation data of blobs and others, we can use the pyBIA data_processing module to
        properly process our data and costruct validation arguments of appropriate shape. 

            >>> blob_x, blob_y = pyBIA.data_processing.process_class(blob_val, label=0, normalize=False)
            >>> other_x, other_x = pyBIA.data_processing.process_class(other_val, label=1, normalize=False)
    
            >>> val_X = np.r_[blob_x, other_x]
            >>> val_Y = np.r_[blob_y, other_y]

            >>> model = create(blob_train, other_train, val_X=val_X, val_Y=val_Y)

        The process_class function will reshape our data and label array, as per the CNN requirements.

        If need-be, this reshaped data array can be constructed manually as follows:

            >>> data = channel.reshape(axis, img_width, img_height, img_num_channels)

        The label array is also a special binary form: you can set this up manually by converting your 1D label arrays 
        into a binary matrix representation. For example, if you have a channel with 100 DIFFUSE samples (label=0), 
        you can create a corresponding label array using numpy and keras:

            >>> label_array = numpy.expand_dims(np.array([0]*len(channel)), axis=1)
            >>> label_array = tensorflow.keras.utils.to_categorical(label, 2)
        
        
    Returns:
        The trained CNN model.
    """
    
    if len(blob_data.shape) != len(other_data.shape):
        raise ValueError("Shape of blob and other data must be the same.")
    if val_X is not None:
        if val_Y is None:
            raise ValueError("Need to input validation data labels (val_Y).")
    if val_Y is not None:
        if val_X is None:
            raise ValueError("Need to input validation data (val_X).")
    if val_X is not None:
        if len(val_X) != len(val_Y):
            raise ValueError("Size of validation data and validation labels must be the same.")
    if batch_size < 16:
        warn("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.", stacklevel=2)
    
    img_width = blob_data[0].shape[0]
    img_height = blob_data[0].shape[1]

    ix = np.random.permutation(len(blob_data))
    blob_data = blob_data[ix]

    ix = np.random.permutation(len(other_data))
    other_data = other_data[ix]

    X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)

    X_train[X_train > 1] = 1
    X_train[X_train < 0] = 0
    input_shape = (img_width, img_height, img_num_channels)
   
    # Uniform scaling initializer
    num_classes = 2
    uniform_scaling = VarianceScaling(
        scale=1.0, mode='fan_in', distribution='uniform', seed=None)

    # Model configuration
    model = Sequential()

    model.add(Conv2D(96, 11, strides=4, activation='relu', input_shape=input_shape,
                     padding=padding, kernel_initializer=uniform_scaling))
    if pooling:
        model.add(MaxPool2D(pool_size=3, strides=2, padding=padding))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(Conv2D(256, 5, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    if pooling:
        model.add(MaxPool2D(pool_size=3, strides=2, padding=padding))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(Conv2D(384, 3, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(384, 3, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(256, 3, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    if pooling:
        model.add(MaxPool2D(pool_size=3, strides=2, padding=padding))
    if batch_norm:
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax',
                    kernel_initializer='TruncatedNormal'))

    optimizer = SGD(learning_rate=lr, momentum=momentum,
                         decay=decay, nesterov=nesterov)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    model_checkpoint = ModelCheckpoint("checkpoint.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [model_checkpoint]

    if val_X is None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1)
    else:
        ix = np.random.permutation(len(val_X))
        val_X, val_Y = val_X[ix], val_Y[ix]
        history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(val_X, val_Y), epochs=epochs, callbacks=callbacks_list, verbose=1)

    path = str(Path.home())+'/'
    print("Saving CNN model and metric files to the local directory.")
    if metrics:
        np.savetxt(path+'model_acc'+filename, history.history['accuracy'])
        np.savetxt(path+'model_loss'+filename, history.history['loss'])
        if val_X is not None:
            np.savetxt(path+'model_val_acc'+filename, history.history['val_accuracy'])
            np.savetxt(path+'model_val_loss'+filename, history.history['val_loss'])

    save_model(model, path+filename+'_Model.h5')

    return model

def predict(data, model, normalize=False, min_pixel=638, max_pixel=3000, target='DIFFUSE', img_num_channels=1):
    """
    Returns the class prediction. The input can either be a single 2D array 
    or a 3D array if there are multiple samples.

    Args:
        data: 2D array for single image, 3D array for multiple images.
        model: The trained Tensorflow model.
        normalize (bool, optional): True will normalize the data using the input min and max pixels
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.
        target (str): The name of the target class, assuming binary classification in 
            which there is an 'OTHER' class. Defaults to 'DIFFUSE'. 
    Returns:
        The class prediction(s), either 'DIFFUSE' or 'OTHER'.

    """
    if data.shape[-1] != 50:
        raise ValueError('Data size is invalid. Each image must be a 50x50 2D array, resize the array \
            using the crop_image function in the data_processing module.')

    data = process_class(data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    predictions = model.predict(data)

    output=[]
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == 0:
            prediction = target
        else:
            prediction = 'OTHER'

        output.append(prediction)

    return np.array(output)

#print("Plotting learning evolution...")
#need subplot to also include loss vs epoch
#plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'])
#plt.xlabel('Epochs')
#plt.ylabel('Training Accuracy')
#plt.title('Model Accuracy')
#plt.show()
