#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:43:16 2021

@author: daniel
"""
import numpy as np

class Preprocess:
	"""
	Parameters
    __________
    channel_1, channel_2, channel_3 : RGB channels

    """
	def __init__(self, channel_1, channel_2=None, channel_3=None)

		if not isinstance(channel_1, np.ndarray):
        	channel_1 = np.array(channel_1)
    	if not isinstance(channel_2, np.ndarray):
            channel_2 = np.array(channel_2)
    	if not isinstance(channel_1, np.ndarray):
            channel_3 = np.array(channel_3)

        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_2 = channel_3


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


def normalize_pixels(channel):
	"""
	Convert pixel data to Unsigned integer (0 to 255)
	and float32, followed by normalization
	"""

	channel = np.array(channel).astype('uint8')
	channel = channel.astype('float32')
	channel /= 255

	return channel

def process_class(channel, label=None, normalize=True):
    """
    Gets a subset of 2D array given a set of (x,y) coordinates
    and an output size. If the slices exceed the bounds 
    of the input array, the non overlapping values
    are filled with NaNs.

    Parameters
    __________
    channel : array
        2D array 
    label : int 
    	Class label
    normalize : bool 
    	True will normalize the data

    Outputs
    _______       
    subset: array
        Subset of the input array
    """
    if normalize:
    	channel = normalize_pixels(channel)

    if len(channel.shape) == 3:
    	img_width = channel[0].shape[0]
    	img_height = channel[0].shape[1]
    	axis = channel.shape[0]
    elif len(channel.shape) == 2:
    	img_width = channel.shape[0]
    	img_height = channel.shape[1]
    	axis = 1
    else:
    	raise ValueError("Channel must either be 2D for a single image or 3D for multiple images.")

    img_num_channels = 1
    data = channel.reshape(axis, img_width, img_height, img_num_channels)

    if label is None:
    	return data

    label = np.expand_dims(np.array([label]*len(channel)), axis=1).astype('uint8')
    return data, label

def create_training_set(blob_data, other_data):
	"""
	Returns image data with corresponding label
	"""
	no_classes = 2
	gb_data, gb_label = process_class(blob, label=0, normalize=True)
	other_data, other_label = process_class(other, label=1, normalize=True)
	training_data = np.r_[gb_data, other_data]
	training_labels = np.r_[gb_label, other_label]

	#one-hot encoding 
	training_labels = np_utils.to_categorical(training_labels, no_classes)

	return training_data, training_labels

