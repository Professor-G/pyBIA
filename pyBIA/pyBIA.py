#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:40:39 2021

@author: daniel
"""
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization

from data_processing import process_class, create_training_set

def pyBIA_model(blob_data, other_data, img_num_channels=1, normalize=True, min_pixel=638, max_pixel=3000, 
    validation_X=None, validation_Y=None, epochs=100, batch_size=32, lr=0.0001, momentum=0.9, decay=0.0005,
    nesterov=False, loss='categorical_crossentropy', padding='same', dropout=0.5, pooling=True, metrics=True, filename=''):
    """
    The CNN model infrastructure presented by AlexNet, with
    modern modifications.
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

    if len(blob_data.shape) == 3: #if matrix is 3D - contains multiple samples
        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]
    else:
        raise ValueError("Data must be 3D, first dimension is number of samples, followed by width and height.")

    X_train, Y_train = create_training_set(blob_data, other_data, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel)
    input_shape = (img_width, img_height, img_num_channels)
   
    # Uniform scaling initializer
    num_classes = 2
    uniform_scaling = VarianceScaling(
        scale=1.0, mode='fan_in', distribution='uniform', seed=None)

    # Model configuration
    model = Sequential()

    model.add(Conv2D(96, 11, strides=4, activation='relu', input_shape=input_shape,
                     padding=padding, kernel_initializer=uniform_scaling))
    if pooling is True:
        model.add(MaxPool2D(pool_size=3, strides=2, padding=padding))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 5, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    if pooling is True:
        model.add(MaxPool2D(pool_size=3, strides=2, padding=padding))
    model.add(BatchNormalization())

    model.add(Conv2D(384, 3, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(384, 3, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(256, 3, activation='relu', padding=padding,
                     kernel_initializer=uniform_scaling))
    if pooling is True:
        model.add(MaxPool2D(pool_size=3, strides=2, padding=padding))
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
    
    if validation_X is None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    elif validation_X is not None:
        history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(validation_X, validation_Y), epochs=epochs, verbose=1)

    if metrics is True:
        np.savetxt('model_acc'+filename, history.history['accuracy'])
        np.savetxt('model_loss'+filename, history.history['loss'])
        if validation_X is not None:
            np.savetxt('model_val_acc'+filename, history.history['val_accuracy'])
            np.savetxt('model_val_loss'+filename, history.history['val_loss'])

    return model


def predict(data, model, normalize=True, min_pixel=638, max_pixel=3000):
    """
    Returns class prediction
    0 for blob
    1 for other
    """
    if len(data.shape) != 4 and normalize is False:
        raise ValueError('Data must be 4 dimensional, use process_class function first.')

    if normalize == True:
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
