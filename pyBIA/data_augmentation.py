#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:28:20 2021

@author: daniel
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_processing import fixed_size_subset
import numpy as np

def generator_parameters():
	"""
	Random image processing 
	paremeters for data augmentation
	"""

	rotation = 180
	width = 0.05
	height = 0.05
	horizontal = True
	vertical = True
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
	"""

	if len(data.shape) == 3:
    		width = data[0].shape[0]
    		height = data[0].shape[1]
	elif len(data.shape) == 2:
    		width = data.shape[0]
    		height = data.shape[1]
	else:
    		raise ValueError("Channel must either be 2D for a single image or 3D for multiple images.")

	resized_images = []
	for i in np.arange(0, len(data)):
    		resized_data = fixed_size_subset(data[i][:, :, 0], width/2., height/2., size)
    		resized_images.append(resized_data)
    
	augmented_data = np.array(data)

	return augmented_data

def augmentation(data, batch_size, resize=True):
	"""
	Performs data augmentation on 
	non-normalized data and 
	resizes image to 50x50
	"""

	rotation, width, height, horizontal, vertical, fill = generator_parameters()[:6]

	datagen = ImageDataGenerator(
        rotation_range=rotation,
        width_shift_range=width,
        height_shift_range=height,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
        fill_mode=fill)


	if len(data.shape) != 4:
		if len(data.shape) == 3:
			data = np.array(np.expand_dims(blob, axis=-1))
		elif len(data.shape) == 2:
			data = np.array(np.expand_dims(blob, axis=-1))
		else:
			raise ValueError("Input data must be 2D for single sample or 3D for multiple sampels")

	augmented_data = []
	for i in np.arange(0, len(data)):
    	original_data = data[i].reshape((1,) + data[-i].shape)
	    for k in range(batch_size):
        	augemented_data = datagen.flow(original_data, batch_size=1)
        	augmented_data.append(augemented_data[0][0])

	augmented_data = np.array(augmented_data)

	if resize == True:
		augmented_data = resize(augmented_data, size=50)

	return augmented_data



