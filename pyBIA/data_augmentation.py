#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:28:20 2021

@author: daniel
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
import numpy as np

from pyBIA.data_processing import fixed_size_subset
from warnings import warn


def resize(data, size=50):
    """
    Resizes the data by cropping out the outer 
    boundaries outside the size x size limit.

    Args:
        data (array): 2D array
        size (int): length/width of the output array

    Returns:
        array: The cropped out data

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


def augmentation(data, batch=10, width_shift=5, height_shift=5, horizontal=True, 
        vertical=True, rotation=0, fill='nearest', image_size=50):
    """
    Performs data augmentation on non-normalized data and resizes image if
    rotational augmentations were applied. 

    Args:
        data (array): 2D array of an image
        batch (int): How many augmented images to create
        width_shift (int): The max shift allowed in either horizontal direction
        height_shift (int): The max shift allowed in either vertical direction
        horizontal (bool): If False no horizontal flips are allowed. Defaults to True.
        vertical (bool): If False no vertical reflections are allowed. Defaults to True.
        rotation (int): The rotation angle in degrees. Defaults to zero for no rotation.
        fill (str) = This is the treatment for data outside the boundaries after roration
            and shifts. Default is set to 'nearest' which repeats the closest pixel values.
            Can set to: {"constant", "nearest", "reflect", "wrap"}.
        image_size (int, bool): The length/width of the cropped image. This can used to remove
            anomalies caused by the fill. Defaults to 50, the pyBIA standard. This can also
            be set to None in which case the image in its original size is returned.

    Note:
        The training set pyBIA uses includes augmented images. The original image size was
        100x100 pixels, these were cropped to 50x50 to remove rotational effects at the 
        outer boundaries. 

    Returns:
        array: 3D array containing the augmented images. 

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
    if augmented_data is not None:
        augmented_data = resize(augmented_data, size=image_size)

    return augmented_data

