#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:43:16 2021

@author: daniel
"""
import numpy as np
from tensorflow.keras.utils import to_categorical

def fixed_size_subset(array, x, y, size):
    """
    Gets a subset of 2D array given a set of (x,y) coordinates
    and an output size. If the slices exceed the bounds 
    of the input array, the non overlapping values
    are filled with NaNs.

    Parameters
    __________
    array: array
        2D array from which to take a subset
    x, y: int 
    	Coordinates of the center of the subset
    size: int 
    	Size of the output array

    Outputs
    _______       
    subset: array
        Subset of the input array
    """

    o, r = np.divmod(size, 2)
    l = (x-(o+r-1)).clip(0)
    u = (y-(o+r-1)).clip(0)

    array_ = array[l: x+o+1, u:y+o+1]
    out = np.full((size, size), np.nan, dtype=array.dtype)
    out[:array_.shape[0], :array_.shape[1]] = array_

    return out

def concat_channels(R, G, B):
    """
    Concatenates three 2D arrays to make a three channel matrix.

    Parameters
    __________
    R, G, B: array
        2D array of the channel

    Outputs
    _______   
    RGB : array
        3D array with each channel stacked.
    """

    if R.ndim != 2 or G.ndim != 2 or B.ndim != 2:
        raise ValueError("Every input channel must be a 2-dimensional array (width + height)")
        
    RGB = (R[..., np.newaxis], G[..., np.newaxis], B[..., np.newaxis])

    return np.concatenate(RGB, axis=-1)


def normalize_pixels(channel, min_pixel=638, max_pixel=7351):
    """
    NDWFS min 0.01% : 638.186
    NDWFS max 99.99% : 7350.639
    """

    channel = np.array(channel).astype('float32')
    channel = (channel - min_pixel)/(max_pixel-min_pixel)

    return channel

def process_class(channel, label=None, normalize=True, min_pixel=638, max_pixel=7351):
    """
    Takes image data from one class, as well as corresponding
    label (0 for blob, 1 for other), and returns the reshaped
    data and the label arrays. This reshaping is required
    for training/testing the classifier
    
    __________
    channel : array
        2D array 
    label : int 
        Class label
    normalize : bool 
        True will normalize the data

    Outputs
    _______       
    data : array
        Reshaped data
    label : array
        Label array

    """

    if normalize:
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

    label = np.expand_dims(np.array([label]*len(channel)), axis=1)
    label = to_categorical(label, 2)
    
    return data, label


def create_training_set(blob_data, other_data, normalize=True, min_pixel=638, max_pixel=7351):
    """
    Returns image data with corresponding label
    """

    gb_data, gb_label = process_class(blob_data, label=0, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel)
    other_data, other_label = process_class(other_data, label=1, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel)
    
    training_data = np.r_[gb_data, other_data]
    training_labels = np.r_[gb_label, other_label]

    return training_data, training_labels


"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

#scaler = MinMaxScaler()

def create_transformation(data, scaler=StandardScaler()):
    
    reshaped_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) 
    transform_fit = scaler.fit(reshaped_data) 
    
    return transform_fit

def transform_images(data, transformation):
    
    reshaped_data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    transformed_data = transformation.transform(reshaped_data)
    
    data = transformed_data.reshape(data.shape[0], data.shape[1], data.shape[2])
    
    return data 
"""

