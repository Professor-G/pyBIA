#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:28:20 2021

@author: daniel
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generator_parameters():
	"""
	Random image processing 
	paremeters
	"""

	rotation = 180
	width = 0.125
	height = 0.125
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

def augmentation(data, batch_size, resize = True):
	"""
	Performs data augmentation on 
	non-normalized data and 
	resizes image to 50x50
	"""

	rotation, width, height, horizontal, vertical, fill = generator_parameters()[:5]

	datagen = ImageDataGenerator(
        rotation_range=rotation,
        width_shift_range=width,
        height_shift_range=height,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
        fill_mode=fill)


	if data.shape != (4,1):
		data = np.array(np.expand_dims(blob, axis=-1))

	augmented_data = []
	for i in np.arange(0, len(data)):
    	original_data = data[i].reshape((1,) + data[-i].shape)
	    for k in range(batch_size):
        	augemented_data = datagen.flow(original_data, batch_size=1)#, target_size=(50, 50))
        	augmented_data.append(augemented_data[0][0])

	augmented_data = np.array(augmented_data)

	if resize == True:
		data = []
		for i in np.arange(0, len(augmented_data)):
    		resized_data = fixed_size_subset(augmented_data[i][:, :, 0], 50, 50, 50)
    		data.append(resized_data)
    
		augmented_data = np.array(data)

	return augmented_data

