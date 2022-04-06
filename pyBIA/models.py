#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:40:39 2021

@author: daniel
"""
import os
import numpy as np
from warnings import warn
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
        was trained using NDWFS blue broadband images. The corresponding .h5 file is 
        located in the data folder inside the pyBIA directory in the Python path. 

    """
    import tensorflow as tf
    from keras.models import load_model
    import pkg_resources

    resource_package = __name__
    resource_path = '/'.join(('data', 'Bw_CNN_Model.h5'))
    model = tf.keras.models.load_model(pkg_resources.resource_filename(resource_package, resource_path))
    
    return model
    
def predict(data, model, normalize=False, min_pixel=638, max_pixel=3000):
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

    Returns:
        array: The class prediction(s), either 'DIFFUSE' or 'OTHER'.

    """

    data = process_class(data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel)
    predictions = model.predict(data)

    output=[]
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == 0:
            prediction = 'DIFFUSE'
        else:
            prediction = 'OTHER'

        output.append(prediction)

    return np.array(output)

def pyBIA_model(blob_data, other_data, img_num_channels=1, normalize=True, 
    min_pixel=638, max_pixel=3000, validation_X=None, validation_Y=None, 
    epochs=100, batch_size=32, lr=0.0001, batch_norm=True, momentum=0.9, decay=0.0005, 
    nesterov=False, loss='categorical_crossentropy', padding='same', 
    dropout=0.5, pooling=True, metrics=True, filename=''):
    """
    The CNN model infrastructure presented by the 2012 ImageNet Large Scale 
    Visual Recognition Challenge, AlexNet. Parameters were adapted for
    our astronomy case of detecting diffuse objects.

    Args:
        blob_data (array): 3D array containing more than one image of diffuse objects.
        other_data (array): 3D array containing more than one image of non-diffuse objects.
        img_num_channels (int): The number of filters used. Defaults to 1, as pyBIA version 1
            has been trained with only blue broadband data.
        normalize (bool, optional): True will normalize the data using the input min and max pixels
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.
        validation_X (array, optional): 3D matrix containing the 2D arrays (images)
            to be used for validation.
        validation_Y (array, optional): A binary class matrix containing the labels of the
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
            zero-value padding around the boundary, ensuring even convolutions.
        dropout (float): Droupout rate between 0 and 1. This is the percentage of dense neurons
            that are turned off at each epoch. This prevents inter-neuron depedency, and thus overfitting. 
            Note: If batch_norm=True, it might be best to disable droput. See: https://arxiv.org/abs/1502.03167
        pooling (bool): True to enable max pooling, false to disable. 
            Note: Max pooling can result in loss of positional information, it computation allows
            setting pooling=False may yield more robust accuracy.
        metrics (bool): When False the CNN will not save the model accuracy and loss. Defaults to True.
        filename (str, optional): The name of the metrics filename. The metrics files will be saved
            as 'model_metric'+filename.

    Note:
        To use a validation dataset when training the model, the validation_X and validation_Y
        parameters must be input. The validation_X is a 3D matrix containing all the images, and
        the validation_Y is another matrix containing their class label (0 for DIFFUSE, 1 for OTHER).

        If you have validation images of blobs and others, we can use the pyBIA data_processing module to
        properly process our data and costruct validation arguments of appropriate shape. 

            >>> val_other, val_other_labels = pyBIA.data_processing.process_class(other_test, label=1, normalize=False)
            >>> val_blob, val_blob_labels = pyBIA.data_processing.process_class(blob_test, label=0, normalize=False)
    
            >>> validation_X = np.r_[val_X1, val_X2]
            >>> validation_Y = np.r_[val_Y1, val_Y2]

            >>> model = pyBIA_model(blob_train, other_train, validation_X=validation_X, validation_Y=validation_Y)

        The process_class function will reshape our data and label array. The data must be a 4D matrix, constructed as follow:

            >>> data = channel.reshape(axis, img_width, img_height, img_num_channels)

        You can do this manually by converting your 1D label arrays into a binary matrix representation.
        For example, if you have a channel with 100 DIFFUSE samples (label=0), you can create a corresponding label array
        with the following, although we recommened you use the process_class function.

            >>> label_array = numpy.expand_dims(np.array([0]*len(channel)), axis=1)
            >>> label_array = tensorflow.keras.utils.to_categorical(label, 2)

    Returns:
        model: The trained Tensorflow model.

    """
    if len(blob_data.shape) != len(other_data.shape):
        raise ValueError("Shape of blob and other data must be the same.")

    if validation_X is not None:
        if validation_Y is None:
            raise ValueError("Need to input validation data labels (validation_Y).")
    if validation_Y is not None:
        if validation_X is None:
            raise ValueError("Need to input validation data (validation_X).")
    if validation_X is not None:
        if len(validation_X) != len(validation_Y):
            raise ValueError("Size of validation data and validation labels must be the same.")

    if batch_size < 16:
        warn("Batch Normalization can be unstable with low batch sizes, if loss returns nan try a larger batch size and/or smaller learning rate.")

    if len(blob_data.shape) == 3: #if matrix is 3D - contains multiple samples
        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]
    else:
        raise ValueError("Data must be 3D, first dimension is number of samples, followed by width and height.")

    X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel)
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

    if validation_X is None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list, verbose=1)
    elif validation_X is not None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(validation_X, validation_Y), epochs=epochs, callbacks=callbacks_list, verbose=1)

    if metrics:
        np.savetxt('model_acc'+filename, history.history['accuracy'])
        np.savetxt('model_loss'+filename, history.history['loss'])
        if validation_X is not None:
            np.savetxt('model_val_acc'+filename, history.history['val_accuracy'])
            np.savetxt('model_val_loss'+filename, history.history['val_loss'])

    save_model(model, filename+'_Model.h5')

    return model

