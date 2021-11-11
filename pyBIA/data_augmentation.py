#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:28:20 2021

@author: daniel
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
import numpy as np

from data_processing import fixed_size_subset
from warnings import warn


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


def augmentation(data, batch=10, width_shift=5, height_shift=5, horizontal=True, vertical=True, rotation=0, fill='nearest', image_size=50):
    """
    Performs data augmentation on 
    non-normalized data and 
    resizes image to 50x50
    """
    if isinstance(width_shift, int) == False or isinstance(height_shift, int) == False or isinstance(rotation, int) == False:
        raise ValueError("Shift parameters must be integers indicating +- pixel range")

    def image_rotation(data):
        return rotate(data, np.random.choice(range(rotation+1), 1)[0], reshape=False, order=0, prefilter=False)
    
    datagen = ImageDataGenerator(
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
        fill_mode=fill)

    if rotation != 0:
        datagen.preprocessing_function = image_rotation

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
    augmented_data = resize(augmented_data, size=image_size)

    return augmented_data

