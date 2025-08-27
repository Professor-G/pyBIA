#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:43:16 2021

@author: daniel
"""
import copy
import numpy as np
from tensorflow.keras.utils import to_categorical

def find_duplicate_features(features, tolerance=1e-9):
    """
    Check for duplicate feature columns using elementwise closeness.

    Parameters
    ----------
    features : ndarray, shape (n_samples, n_features)
        Feature matrix with features stored column-wise.
    tolerance : float, optional
        Absolute tolerance for equality checks passed to np.isclose. Default is 1e-9.

    Returns
    -------
    set of int
        Zero-based indices of columns identified as duplicates of at least one other column within the tolerance; empty if none.
    """

    # Initialize a set to store the unique feature indices
    unique_indices = set()

    # Initialize a set to store the duplicate feature indices
    duplicate_indices = set()

    # Get the transpose of the features array
    features_T = features.T

    # Get the number of columns
    num_cols = features_T.shape[0]
    for i in range(num_cols):
        column1 = features_T[i]
        for j in range(i+1, num_cols):
            column2 = features_T[j]
            if np.all(np.isclose(column1, column2, atol=tolerance)):
                if i not in unique_indices:
                    unique_indices.add(i)
                    duplicate_indices.add(i)
                if j not in unique_indices:
                    unique_indices.add(j)
                    duplicate_indices.add(j)

    return duplicate_indices

def crop_image(data, x, y, size=50, invert=False):
    """
    Return a square sub-array of length `size` centered on (x, y) from a 2D image, padding with NaNs if the crop extends beyond the image bounds.

    Parameters
    ----------
    data : ndarray
        Input 2D image array.
    x : int
        Column (x) coordinate of the crop center, in the image's coordinate convention.
    y : int
        Row (y) coordinate of the crop center, in the image's coordinate convention.
    size : int, optional
        Width/height of the output crop in pixels; must be a positive integer. Default is 50.
    invert : bool, optional
        If True, swap the provided (x, y) before cropping (useful when inputs come from FITS-style top-left origins but the cropper assumes standard indexing). Default is False.

    Returns
    -------
    ndarray
        Cropped array of shape (size, size); regions falling outside the input image are padded with NaNs.
    """

    if invert:
        x, y = y, x
        
    data_copy = copy.deepcopy(data)

    o, r = np.divmod(size, 2)
    l = (int(x)-(o+r-1)).clip(0)
    u = (int(y)-(o+r-1)).clip(0)
    array = data_copy[l: int(x)+o+1, u:int(y)+o+1]
    
    out = np.full((size, size), np.nan, dtype=data_copy.dtype)
    out[:array.shape[0], :array.shape[1]] = array

    return out

def concat_channels(channel1, channel2, channel3=None):
    """
    Concatenate up to three single-band 2D images along a new last axis, producing a multi-channel tensor.

    Parameters
    ----------
    channel1 : ndarray
        2D array for the first channel (H × W). Must have the same height and width as the other channels.
    channel2 : ndarray
        2D array for the second channel (H × W). Must have the same height and width as the other channels.
    channel3 : ndarray or None, optional
        2D array for the third channel (H × W). Must have the same height and width as the other channels. Default is None.

    Returns
    -------
    ndarray
        3D array of shape (H, W, C) where C is 2 if `channel3` is None, otherwise 3. The dtype matches the input arrays.
    """
    
    if channel3 is None:
        colorized = (channel1[..., np.newaxis], channel2[..., np.newaxis])
    else:
        colorized = (channel1[..., np.newaxis], channel2[..., np.newaxis], channel3[..., np.newaxis])

    return np.concatenate(colorized, axis=-1)

def normalize_pixels(channels, min_pixel, max_pixel, img_num_channels):
    """
    Clip and min–max normalize image data per channel into [0, 1], handling 2D, 3D, and 4D inputs.

    Parameters
    ----------
    channels : ndarray
        Input image data as (H, W), (N, H, W), (H, W, C), or (N, H, W, C); non-finite values are set to `min_pixel`.
    min_pixel : float
        Lower clip bound applied to all channels before normalization.
    max_pixel : float or list of float
        Upper clip bound; a scalar for single-channel data or a list of length `img_num_channels` for multi-channel data.
    img_num_channels : int
        Number of channels expected in the output; used to validate or reshape inputs.

    Returns
    -------
    ndarray
        Normalized array with values in [0, 1]; shape is (N, H, W) for single-channel inputs or (N, H, W, C) for multi-channel inputs.

    Raises
    ------
    ValueError
        If `max_pixel` type is incompatible with `img_num_channels`, if shapes are inconsistent with `img_num_channels`,
        or if the input dimensionality is not 2D, 3D, or 4D.
    """

    if isinstance(max_pixel, int) and img_num_channels != 1:
        raise ValueError('The max_pixel parameter should be a list containing the value for each band!')
    if isinstance(max_pixel, int) is False and img_num_channels == 1:
        if isinstance(max_pixel, list):
            max_pixel = max_pixel[0]
        else:
            raise ValueError('If img_num_channels is 1 the max_pixel input must be an integer/float or list.')

    images = copy.deepcopy(channels)

    #The min pixel replaces NaN and below threshold values.
    images[np.isfinite(images) == False] = min_pixel 
    images[images < min_pixel] = min_pixel

    if img_num_channels == 1:
        images[images > max_pixel] = max_pixel
        return (images - min_pixel) /  (max_pixel - min_pixel)

    #Setting array dimensions for consistency#
    if len(images.shape) == 4:
        axis = images.shape[0]
        if images.shape[-1] != img_num_channels:
            raise ValueError('img_num_channels parameter must match the number of filters! Number of filters detected: '+str(channels.shape[-1]))
        img_width, img_height = images[0].shape[1], images[0].shape[0]
    elif len(images.shape) == 3:
        if img_num_channels == 1:
            axis, img_width, img_height = images.shape[0], images.shape[1],images.shape[2]
        else:
            axis, img_width, img_height = 1, images.shape[0], images.shape[1]
    elif len(images.shape) == 2:
        axis, img_width, img_height = 1, images.shape[1], images.shape[0]
    else:
        raise ValueError("Channel must either be 2D for a single sample, 3D for multiple samples or single sample with multiple filters, or 4D for multifilter images.")

    images = images.reshape(axis, img_width, img_height, img_num_channels)

    for i in range(img_num_channels):
        images[:,:,:,i][images[:,:,:,i] > max_pixel[i]] = max_pixel[i]
        images[:,:,:,i] = (images[:,:,:,i] - min_pixel) /  (max_pixel[i] - min_pixel)

    return images 

def process_class(channel, label=None, img_num_channels=1, normalize=True, min_pixel=638, max_pixel=3000):
    """
    Reshape image data to (N, H, W, C) and optionally apply per-channel min–max normalization; optionally return one-hot labels.

    Parameters
    ----------
    channel : ndarray
        Input images as (H, W) for one image, (N, H, W) for many images, (H, W, C) for one multi-channel image, or (N, H, W, C) for many multi-channel images.
    label : int or None, optional
        Class label encoded as 0 or 1; if provided, a one-hot label array is returned alongside the data; Default is None.
    img_num_channels : int, optional
        Number of channels per sample used to validate and reshape the output; Default is 1.
    normalize : bool, optional
        If True, clip to [`min_pixel`, `max_pixel`] and scale each channel to [0, 1] using `normalize_pixels`; Default is True.
    min_pixel : float, optional
        Lower clip bound applied when `normalize=True`, with non-finite values also set to this bound; Default is 638.
    max_pixel : float or list of float, optional
        Upper clip bound(s) applied when `normalize=True`, scalar for single-channel or list of length `img_num_channels` for multi-channel; Default is 3000.

    Returns
    -------
    ndarray
        Data array shaped (N, H, W, C) with values in [0, 1] if normalized, otherwise an unscaled copy.
    ndarray
        One-hot label array shaped (N, 2) returned only when `label` is not None.
    """

    if normalize:
        if len(channel) >= 1000:
            print('Normalizing images...') #For when predictions are being made
        data = normalize_pixels(channel, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    else:
        images = copy.deepcopy(channel)
        if len(images.shape) == 4:
            axis = images.shape[0]
            if images.shape[-1] != img_num_channels:
                raise ValueError('img_num_channels parameter must match the number of filters! Number of filters detected: '+str(channel.shape[-1]))
            img_width = images[0].shape[1]
            img_height = images[0].shape[0]
        elif len(images.shape) == 3:
            if img_num_channels == 1 :
                axis = images.shape[0]
                img_width = images.shape[1]
                img_height = images.shape[2]
            else:
                axis = 1
                img_width = images.shape[0]
                img_height = images.shape[1]
        elif len(images.shape) == 2:
            img_width = images.shape[1]
            img_height = images.shape[0]
            axis = 1
        else:
            raise ValueError("Channel must either be 2D for a single sample, 3D for multiple samples or single sample with multiple filters, or 4D for multifilter images.")
        data = images.reshape(axis, img_width, img_height, img_num_channels)
    
    if label is None:
        return data

    #reshape
    label = np.expand_dims(np.array([label]*len(data)), axis=1)
    label = to_categorical(label, 2)
    
    return data, label

def create_training_set(blob_data, other_data, img_num_channels=1, normalize=True, min_pixel=638, max_pixel=3000):
    """
    Combine positive and negative image stacks into a single training tensor with one-hot labels.

    Parameters
    ----------
    blob_data : ndarray
        Positive-class images shaped (N, H, W) or (N, H, W, C) with C = img_num_channels; these receive label 1.
    other_data : ndarray
        Negative-class images shaped (N, H, W) or (N, H, W, C) with C = img_num_channels; these receive label 0.
    img_num_channels : int, optional
        Number of channels per sample used to validate/reshape the output; Default is 1.
    normalize : bool, optional
        If True, clip to [min_pixel, max_pixel] and scale per channel to [0, 1] using `normalize_pixels`; Default is True.
    min_pixel : float, optional
        Lower clip bound applied when `normalize` is True; non-finite values are also set to this bound; Default is 638.
    max_pixel : float or list of float, optional
        Upper clip bound(s) applied when `normalize` is True; scalar for single-channel or list of length `img_num_channels` for multi-channel; Default is 3000.

    Returns
    -------
    ndarray
        Training images shaped (N_total, H, W, C) with C = img_num_channels; values in [0, 1] when `normalize` is True.
    ndarray
        One-hot labels shaped (N_total, 2) with class 1 for `blob_data` and class 0 for `other_data`.

    Notes
    -----
    This function is for binary classification only; for multi-class workflows, call `process_class` per class and concatenate the results.
    """

    class1_data, class1_label = process_class(blob_data, label=1, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    class2_data, class2_label = process_class(other_data, label=0, normalize=normalize, min_pixel=min_pixel, max_pixel=max_pixel, img_num_channels=img_num_channels)
    
    training_data = np.r_[class1_data, class2_data]
    training_labels = np.r_[class1_label, class2_label]

    return training_data, training_labels

# Log transformation for the Hu moments
def signed_log_transform(x, eps=1e-12):
    """
    Apply a signed base-10 logarithmic transform that preserves the sign of each value.

    This is useful for features spanning several orders of magnitude (e.g., Hu moments)
    that can be positive or negative; zeros remain zero due to the sign factor.

    Parameters
    ----------
    x : array-like or scalar
        Input value(s) to transform.
    eps : float, optional
        Small positive constant added inside the log to avoid log(0); Default is 1e-12.

    Returns
    -------
    ndarray or scalar
        Transformed value(s) with the same shape as `x`.
    """
    return np.sign(x) * np.log10(np.abs(x) + eps)
