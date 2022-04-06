#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:43:16 2021

@author: daniel
"""
import numpy as np
from tensorflow.keras.utils import to_categorical

def crop_image(data, x, y, size):
    """
    This function takes a 2D array and returns a sub-array
    centered around x and y. 

    The sub array will be a square of length = size.

    Note:
        When applying data augmentation techniques 
        it is best to crop the image afterward, to
        avoid the rotational shear visible on the edges.

    Args:
        data (array): 2D array.
        x, y (int): Central position of the sub-array to be cropped, 
        relative to the entire data.
        size (int): length/width of the output array.

    Returns:
        array: The cropped array.

    """
    o, r = np.divmod(size, 2)
    l = (int(x)-(o+r-1)).clip(0)
    u = (int(y)-(o+r-1)).clip(0)
    array = data[l: int(x)+o+1, u:int(y)+o+1]
    
    out = np.full((size, size), np.nan, dtype=data.dtype)
    out[:array.shape[0], :array.shape[1]] = array

    return out

def concat_channels(channel1, channel2, channel3):
    """
    This function concatenates three 2D arrays to make a three channel matrix.
    Useful for image classification when using multiple filters.

    Can combine SDSS g,r,i for example, to make one 3D image. Order at which
    they are stacked must be conistent if data is input for classification.

    Note:
        pyBIA version 1 contains a trained CNN model that made use of
        only a single band, although it is possible to also train
        a CNN model using multiple bands if the data availability allows.
    
    Args:
        Channel1 (array): 2D array of the first channel.
        Channel2 (array): 2D array of the second channel.
        Channel3 (array): 2D array of the third channel.

    Returns:
        array: 3D array with each channel stacked.

    """

    if channel1.ndim != 2 or channel2.ndim != 2 or channel3.ndim != 2:
        raise ValueError("Every input channel must be a 2-dimensional array (width + height)")
        
    colorized = (channel1[..., np.newaxis], channel2[..., np.newaxis], channel3[..., np.newaxis])

    return np.concatenate(colorized, axis=-1)


def normalize_pixels(channel, min_pixel=638, max_pixel=3000):
    """
    This function will apply min-max normalization. 

    NDWFS min 0.01% : 638.186

    NDWFS max 99.99% : 7350.639

    Max intensity of expected blobs : ~3000

    Args:
        channel (array): 2D array for one image, 3D array for multiple images.
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.

    Returns:      
        array: Reshaped data and label arrays.

    """
        
    channel = (channel - min_pixel) /  (max_pixel - min_pixel)

    return channel

def process_class(channel, label=None, normalize=True, min_pixel=638, max_pixel=3000):
    """
    Takes image data and returns the reshaped data array, which is required when 
    entering data into the CNN classifier.
    
    If label is set to either 0 or 1, then the reshaped data is
    returned along with an array containing the label array, also reshaped. 
    This is used for creating training or validations sets of appropriate shape.
    
    Note:
        Image anomalies can be removed by setting normalize=True, as the 
        values below/above the thresholds are set to the min/max limits. We
        strongly recommend normalizing your data.

    Args:
        channel (array): 2D array for one image, 3D array for multiple images.
        label (int, optional): Class label, 0 for blob, 1 for other. Defaults to None.
        normalize (bool, optional): True will apply min-max normalization.
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.

    Returns:      
        array: Reshaped data and label arrays

    """

    if normalize is True:
        channel[np.isnan(channel) == True] = min_pixel 
        channel[channel > max_pixel] = max_pixel
        channel[channel < min_pixel] = min_pixel
        channel = normalize_pixels(channel, min_pixel=min_pixel, max_pixel=max_pixel)

    if len(channel.shape) == 3:
        img_width = channel[0].shape[0]
        img_height = channel[0].shape[1]
        axis = channel.shape[0]
    elif len(channel.shape) == 2:
        img_width = channel.shape[0]
        img_height = channel.shape[1]
        axis = 1
    else:
        raise ValueError("Channel must either be 2D for a single sample or 3D for multiple samples.")

    img_num_channels = 1
    data = channel.reshape(axis, img_width, img_height, img_num_channels)

    if label is None:
        #warn("Returning processed data only, as no corresponding label was input.")
        return data

    #reshape
    label = np.expand_dims(np.array([label]*len(channel)), axis=1)
    label = to_categorical(label, 2)
    
    return data, label


def create_training_set(blob_data, other_data, normalize=True, min_pixel=638, max_pixel=3000):
    """
    Combines image data of known class to create a training set.
    This is used for training the machine learning models. 

    Args:
        blob_data (array): 3D array containing more than one image of diffuse objects.
        other_data (array): 3D array containing more than one image of non-diffuse objects.
        normalize (bool, optional): True will normalize the data using the input min and max pixels
        min_pixel (int, optional): The minimum pixel count, defaults to 638. 
            Pixels with counts below this threshold will be set to this limit.
        max_pixel (int, optional): The maximum pixel count, defaults to 3000. 
            Pixels with counts above this threshold will be set to this limit.

    Returns:      
        array: Reshaped data and label arrays.

    """

    gb_data, gb_label = process_class(blob_data, label=0, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel)
    other_data, other_label = process_class(other_data, label=1, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel)
    
    training_data = np.r_[gb_data, other_data]
    training_labels = np.r_[gb_label, other_label]

    return training_data, training_labels


