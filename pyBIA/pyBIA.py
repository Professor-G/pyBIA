#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 22:40:39 2021

@author: daniel
"""
import os
import numpy as np

from keras.models import Sequential
from keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization

from data_processing import process_class, create_training_set

def hyperparameters():
    """
    Sets the hyperparemeters to be used when 
    constructing the convolutional NN.
    """
    epochs=100
    batch_size=32
    learning_rate=0.0001
    momentum=0.9
    decay=0.0005
    nesterov=False
    loss='categorical_crossentropy'
    return epochs, batch_size, learning_rate, momentum, decay, nesterov, loss


def pyBIA_model(blob_data, other_data, img_num_channels=1):
    """
    The CNN model infrastructure presented by AlexNet, with
    modern modifications
    """
    if len(blob_data.shape) != len(other_data.shape):
        raise ValueError("Shape of blob and other data must be the same.")


    if len(blob_data.shape) == 3: #if matrix is 3D - contains multiple sampels
        img_width = blob_data[0].shape[0]
        img_height = blob_data[0].shape[1]

    elif len(blob_data.shape) == 2: #if matrix is 2D - contains one sample
        img_width = blob_data.shape[0]
        img_height = blob_data.shape[1]

    X_train, Y_train = create_training_set(blob_data, other_data)
    input_shape = (img_width, img_height, img_num_channels)

    epochs, batch_size, lr, momentum, decay, nesterov, loss = hyperparameters()
   
    # Uniform scaling initializer
    num_classes = 2
    uniform_scaling = VarianceScaling(
        scale=1.0, mode='fan_in', distribution='uniform', seed=None)

    # Model configuration
    model = Sequential()

    model.add(Conv2D(96, 11, strides=4, activation='relu', input_shape=input_shape,
                     padding='same', kernel_initializer=uniform_scaling))
    #model.add(MaxPool2D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 5, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    #model.add(MaxPool2D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(384, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(384, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(256, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    #model.add(MaxPool2D(pool_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax',
                    kernel_initializer='TruncatedNormal'))

    optimizer = SGD(learning_rate=learning_rate, momentum=momentum,
                         decay=decay, nesterov=nesterov)

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=no_epochs, verbose=1)

    return model


def predict(data, model, normalize=True):
    """
    Returns class prediction
    0 for blob
    1 for other
    """
    if normalize == True:
        data = process_class(data)

    pred = model.predict(data)
    pred = np.argmax(pred)

    return pred
