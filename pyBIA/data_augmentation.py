#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:28:20 2021

@author: daniel
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_processing import fixed_size_subset
import numpy as np
from warnings import warn

def generator_parameters():
    """
    Random image processing 
    paremeters for data augmentation
    """
    rotation = 0
    width = 0.0
    height = 0.0
    horizontal = False
    vertical = False
    fill = 'nearest'
    cval = 0.0
    zca = False
    zca_epsilon = 1e-06
    brightness = None
    shear = 0.0
    zoom = 0.0
    rescale = None
    sample_norm = False
    feature_norm = False
    std_norm = False
    sample_std = False
    data_format = None
    split = 0.0
    return rotation, width, height, horizontal, vertical, fill, cval
    zca, zca_epsilon, brightness, shear, zoom, rescale, sample_norm,
    feature_norm, std_norm, sample_std, data_format, split

def resize(data, size=50):
    """
    Resize image
    size x size
    """
    if len(data.shape) == 3 or len(data.shape) == 4:
        width = data[0].shape[0]
        height = data[0].shape[1]
    elif len(data.shape) == 2:
        width = data.shape[0]
        height = data.shape[1]
    else:
        raise ValueError("Channel cannot be one dimensional")

    if width != height:
        raise ValueError("Can only resize square images")
    if width == size:
        warn("No resizing necessary, image shape is already in desired size")
        if len(data.shape) == 4:
            data = data[:, :, :, 0]
        return data

    if len(data.shape) == 2:
        resized_data = fixed_size_subset(np.array(np.expand_dims(data, axis=-1))[:, :, 0], int(width/2.), int(height/2.), size)
        return resized_data
    else:
        resized_images = []    
        for i in np.arange(0, len(data)):
            if len(data[i].shape) == 2:
                resized_data = fixed_size_subset(np.array(np.expand_dims(data[i], axis=-1))[:, :, 0], int(width/2.), int(height/2.), size)
            else:
                resized_data = fixed_size_subset(data[i][:, :, 0], int(width/2.), int(height/2.), size)
            resized_images.append(resized_data)

    resized_data = np.array(resized_images)
    return resized_data

def augmentation(data, batch=10, image_width=50, rotation=180, width=0.05, height=0.05, horizontal=True, vertical=True, fill='nearest'):
    """
    Performs data augmentation on 
    non-normalized data and 
    resizes image to 50x50
    """

    datagen = ImageDataGenerator(
        rotation_range=rotation,
        width_shift_range=width,
        height_shift_range=height,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
        fill_mode=fill)

    if len(data.shape) != 4:
        if len(data.shape) == 3 or len(data.shape) == 2:
            data = np.array(np.expand_dims(data, axis=-1))
        else:
            raise ValueError("Input data must be 2D for single sample or 3D for multiple samples")

    augmented_data = []
    for i in np.arange(0, len(data)):
        original_data = data[i].reshape((1,) + data[-i].shape)
        for k in range(batch):
        	augement = datagen.flow(original_data, batch_size=1)
        	augmented_data.append(augement[0][0])

    augmented_data = np.array(augmented_data)
    augmented_data = resize(augmented_data, size=image_width)

    return augmented_data

