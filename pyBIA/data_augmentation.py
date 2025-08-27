#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:28:20 2021

@author: daniel
"""
from pyBIA.data_processing import crop_image, concat_channels
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from warnings import warn
import numpy as np
import random
from typing import Optional
from scipy.ndimage import affine_transform

def augmentation(
    channel1, 
    channel2=None, 
    channel3=None, 
    batch=1, 
    width_shift=0, 
    height_shift=0, 
    horizontal=False, 
    vertical=False, 
    rotation=False, 
    fill='nearest', 
    image_size=None, 
    zoom_range=None, 
    mask_size=None, 
    num_masks=None, 
    blend_multiplier=0, 
    blending_func='mean', 
    num_images_to_blend=2, 
    skew_angle=0, 
    return_stacked=False
    ):
    """
    Offline image augmentation for up to three aligned channels (bands), with optional stacking.

    Parameters
    ----------
    channel1 : ndarray
        First channel as a single image (H×W) or a stack (N×H×W). Always required.
    channel2 : ndarray or None, optional
        Second channel aligned to `channel1`, shape (H×W) or (N×H×W). Default is None.
    channel3 : ndarray or None, optional
        Third channel aligned to `channel1`, shape (H×W) or (N×H×W). Default is None.
    batch : int, optional
        Number of augmented samples to generate per input image. Default is 1.
    width_shift : int, optional
        Maximum horizontal pixel shift (both directions). Default is 0.
    height_shift : int, optional
        Maximum vertical pixel shift (both directions). Default is 0.
    horizontal : bool, optional
        Random left–right flips if enabled. Default is False.
    vertical : bool, optional
        Random up–down flips if enabled. Default is False.
    rotation : bool, optional
        Random rotation by an angle uniformly sampled in [0°, 360°] if enabled. Default is False.
    fill : {'constant','nearest','reflect','wrap'}, optional
        Fill mode used for pixels introduced by shifts/rotations. Default is 'nearest'.
    image_size : int or None, optional
        Output side length for cropping/resizing after transforms (returns H=W=image_size); if None, keep original size. Default is None.
    zoom_range : tuple[float, float] or None, optional
        (min_zoom, max_zoom) factor applied uniformly at random (e.g., (0.9, 1.1)); if None, no zoom is applied. Default is None.
    mask_size : int or None, optional
        Diameter of circular cutout mask applied at random locations; if None, cutouts are disabled. Default is None.
    num_masks : int or None, optional
        Number of cutouts per image when `mask_size` is set; requires `mask_size`. Default is None.
    blend_multiplier : float, optional
        Ratio controlling the number of synthetic blended images (≥1 replaces/expands the set, <1 disables blending); 0 disables entirely. Default is 0.
    blending_func : {'mean','max','min','random'}, optional
        Reduction used when blending multiple images into one. Default is 'mean'.
    num_images_to_blend : int, optional
        Number of images randomly selected to compose each blend when blending is enabled. Default is 2.
    skew_angle : float, optional
        Maximum absolute skew angle in degrees (sampled uniformly from [-skew_angle, +skew_angle]); 0 disables skew. Default is 0.
    return_stacked : bool, optional
        If True, return channels concatenated along a last dimension (N×H×W×C); otherwise return one array per input channel. Default is False.

    Returns
    -------
    aug1 : ndarray
        Augmented images for `channel1`, shape (M×H×W) when not stacked, where M = N×batch possibly modified by zoom/blend steps.
    aug2 : ndarray, optional
        Augmented images for `channel2` when provided and `return_stacked` is False, same length and shape as `aug1`.
    aug3 : ndarray, optional
        Augmented images for `channel3` when provided and `return_stacked` is False, same length and shape as `aug1`.
    stacked : ndarray
        If `return_stacked` is True, a single array of shape (M×H×W×C) with C equal to the number of provided channels (2 or 3).

    Notes
    -----
    - Identical random seeds are reused across channels so that all spatial transforms (shift/flip/rotate/zoom/skew, blending, cutouts) are aligned.
    - If `mask_size` is provided, `num_masks` must also be provided (and vice-versa).
    - `width_shift` and `height_shift` must be integers specifying pixel ranges.
    - Setting `blend_multiplier >= 1` generates blended samples; e.g., 1.0 replaces with blends, 1.5 increases the set by 50% using blends.
    """

    if batch == 0: #Setting this in case the negative class is not set to be augmented during the CNN optimization routine.
        if channel2 is None:
            return channel1 
        else:
            if channel3 is None:
                return channel1, channel2
            else:
                return channel1, channel2, channel3

    if isinstance(width_shift, int) == False or isinstance(height_shift, int) == False:
        raise ValueError("Shift parameters must be integers indicating +- pixel range")
    if mask_size is not None:
        if num_masks is None:
            raise ValueError('Need to input num_masks parameter.')
    if num_masks is not None:
        if mask_size is None:
            raise ValueError('Need to input mask_size parameter.')

    if rotation:
        rotation = 360
    else:
        rotation = 0

    def image_rotation(data):
        """
        Function for the image data genereation which hardcodes the rotation parameter of the parent function.
        The order parameter to 0 to ensure that the rotation is performed using nearest-neighbor interpolation, 
        which minimizes the amount of distortion introduced into the image. Additionally, the reshape parameter is False, 
        which will ensure that the rotated image has the same shape as the original image. 
        """
        return rotate(data, np.random.choice(range(rotation+1), 1)[0], reshape=False, order=0, prefilter=True) #Prefilter is useful but slows things down slightly
        
    #Tensorflow Image Data Generator with shifts and flips. While fill is also an option, in practice it is best to simply crop an oversized image 
    datagen = ImageDataGenerator(
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        horizontal_flip=horizontal,
        vertical_flip=vertical,
        fill_mode=fill
        )

    #The rotation child function is only added to the Image Data Generator if rotation parameter is input
    if rotation != 0:
        datagen.preprocessing_function = image_rotation

    if len(channel1.shape) == 3: 
        data = np.array(np.expand_dims(channel1, axis=-1))
    elif len(channel1.shape) == 2:
        data = np.array(np.expand_dims(channel1, axis=-1))
        data = data.reshape((1,) + data.shape)
    else:
        raise ValueError("Input data must be 2D for single sample or 3D for multiple samples")

    augmented_data, seeds = [], [] #Seeds will store the rotation/translation/shift and/or zoom augmentations for multi-band reproducibility
    for i in np.arange(0, len(data)):
        original_data = data[i].reshape((1,) + data[-i].shape)
        for j in range(batch):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds.append(seed)
           # import pdb; pdb.set_trace()
            augment = datagen.flow(original_data, batch_size=1, seed=seed) #returns 3D (width, height, num)
            augmented_data_batch = augment.__next__()[0]
            width, height = augmented_data_batch.shape[:2]
            augmented_data_reshaped = np.reshape(augmented_data_batch, (width, height))
            if zoom_range is not None:
                augmented_data.append(resize(random_zoom(augmented_data_reshaped, zoom_min=zoom_range[0], zoom_max=zoom_range[1], seed=seed), image_size))
            else:
                augmented_data.append(resize(augmented_data_reshaped, image_size))

    augmented_data = np.array(augmented_data)

    if skew_angle != 0:
        seeds_skew, a = [], [] #Individual images will be input independently to the a list to ensure proper seed use
        for i in range(len(augmented_data)):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds_skew.append(seed)
            a.append(random_skew(augmented_data[i], max_angle=skew_angle, seed=seed))
        augmented_data = np.array(a)

    if blend_multiplier >= 1:
        seeds_blend, a = [], [] #Individual images will be input independently to a list along with the seed to ensure reproducibility across all channels
        for i in range(int(blend_multiplier*len(augmented_data))):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds_blend.append(seed)
            blended_image = image_blending(augmented_data, num_augmentations=1, blending_func=blending_func, num_images_to_blend=num_images_to_blend, seed=seed)
            a.append(blended_image[0])
        augmented_data = np.array(a)

    if mask_size is not None:
        seeds_mask, a = [], [] #Individual images will be input independently to the a list to ensure proper seed use
        for i in range(len(augmented_data)):
            seed = int(random.sample(range(int(1e9)), 1)[0])
            seeds_mask.append(seed)
            a.append(random_cutout(augmented_data[i], mask_size=mask_size, num_masks=num_masks, seed=seed))
        augmented_data = np.array(a)

    if channel2 is None:
        return augmented_data
    else:
        if len(channel2.shape) == 3: 
            data = np.array(np.expand_dims(channel2, axis=-1))
        elif len(channel2.shape) == 2:
            data = np.array(np.expand_dims(channel2, axis=-1))
            data = data.reshape((1,) + data.shape)
     
        augmented_data2, k = [], 0
        for i in np.arange(0, len(data)):
            original_data = data[i].reshape((1,) + data[-i].shape)
            for j in range(batch):
                augment = datagen.flow(original_data, batch_size=1, seed=seeds[k])
                augmented_data_batch = augment.__next__()[0]
                width, height = augmented_data_batch.shape[:2]
                augmented_data_reshaped = np.reshape(augmented_data_batch, (width, height))
                if zoom_range is not None:
                    augmented_data2.append(resize(random_zoom(augmented_data_reshaped, zoom_min=zoom_range[0], zoom_max=zoom_range[1], seed=seeds[k]), image_size))
                else:
                    augmented_data2.append(resize(augmented_data_reshaped, image_size))
                k += 1

    augmented_data2 = np.array(augmented_data2)

    if skew_angle != 0:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data2)):
            a.append(random_skew(augmented_data2[i], max_angle=skew_angle, seed=seeds_skew[i]))
        augmented_data2 = np.array(a)

    if blend_multiplier >= 1:
        a = [] #Individual images will be input independently to a list to ensure reproducibility across all channels
        for i in range(int(blend_multiplier*len(augmented_data2))):
            blended_image = image_blending(augmented_data2, num_augmentations=1, blending_func=blending_func, num_images_to_blend=num_images_to_blend, seed=seeds_blend[i])
            a.append(blended_image[0])
        augmented_data2 = np.array(a)

    if mask_size is not None:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data2)):
            a.append(random_cutout(augmented_data2[i], mask_size=mask_size, num_masks=num_masks, seed=seeds_mask[i]))
        augmented_data2 = np.array(a)

    if channel3 is None:
        if return_stacked:
            return concat_channels(augmented_data, augmented_data2)
        else:
            return augmented_data, augmented_data2
    else:
        if len(channel3.shape) == 3: 
            data = np.array(np.expand_dims(channel3, axis=-1))
        elif len(channel3.shape) == 2:
            data = np.array(np.expand_dims(channel3, axis=-1))
            data = data.reshape((1,) + data.shape)
   
        augmented_data3, k = [], 0
        for i in np.arange(0, len(data)):
            original_data = data[i].reshape((1,) + data[-i].shape)
            for j in range(batch):
                augment = datagen.flow(original_data, batch_size=1, seed=seeds[k])
                augmented_data_batch = augment.__next__()[0]
                width, height = augmented_data_batch.shape[:2]
                augmented_data_reshaped = np.reshape(augmented_data_batch, (width, height))
                if zoom_range is not None:
                    augmented_data3.append(resize(random_zoom(augmented_data_reshaped, zoom_min=zoom_range[0], zoom_max=zoom_range[1], seed=seeds[k]), image_size))
                else:
                    augmented_data3.append(resize(augmented_data_reshaped, image_size))
                k += 1

    augmented_data3 = np.array(augmented_data3)

    if skew_angle != 0:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data3)):
            a.append(random_skew(augmented_data3[i], max_angle=skew_angle, seed=seeds_skew[i]))
        augmented_data3 = np.array(a)

    if blend_multiplier >= 1:
        a = [] #Individual images will be input independently to a list to ensure reproducibility across all channels
        for i in range(int(blend_multiplier*len(augmented_data3))):
            blended_image = image_blending(augmented_data3, num_augmentations=1, blending_func=blending_func, num_images_to_blend=num_images_to_blend, seed=seeds_blend[i])
            a.append(blended_image[0])
        augmented_data3 = np.array(a)

    if mask_size is not None:
        a = [] #Individual images will be input independently to ensure proper seed use
        for i in range(len(augmented_data3)):
            a.append(random_cutout(augmented_data3[i], mask_size=mask_size, num_masks=num_masks, seed=seeds_mask[i]))
        augmented_data3 = np.array(a)

    if return_stacked:
        return concat_channels(augmented_data, augmented_data2, augmented_data3)
    else:
        return augmented_data, augmented_data2, augmented_data3

def random_cutout(
    images, 
    mask_size=16, 
    num_masks=1, 
    seed=None, 
    mask_type='circle'
    ):
    """
    Cutout augmentation: randomly applies square or circular zeroed masks to each image.

    Parameters
    ----------
    images : ndarray
        Input as a single image (H×W) or a stack (N×H×W); values are preserved except where masked. Default is required.
    mask_size : int, optional
        Mask scale: radius for 'circle'; half-side for 'square' (final square = 2×mask_size by 2×mask_size). Default is 16.
    num_masks : int, optional
        Number of masks to place per image; masks may overlap. Default is 1.
    seed : int or None, optional
        Random seed for reproducible mask placement; if None, use current RNG state. Default is None.
    mask_type : {'square','circle'}, optional
        Shape of each mask region applied to the image(s). Default is 'circle'.

    Returns
    -------
    ndarray
        Array with cutouts applied; same shape as input (H×W for a single image, N×H×W for a stack).
    """

    if seed is not None:
        np.random.seed(seed)

    if images.ndim == 3:
        num_images, height, width = images.shape
    elif images.ndim == 2:
        height, width = images.shape
        num_images = 1
    else:
        raise ValueError('Input array must be either 2D (single image) or 3D (multiple images)')

    #Reshape input from (num_images, height, width) to (num_images, height, width, 1)
    images = images.reshape(-1, height, width, 1)

    new_images = np.copy(images)

    for i in range(num_images):
        for j in range(num_masks):
            if mask_type == 'square':
                if height - 2*mask_size > 0 and width - 2*mask_size > 0:
                    h = np.random.randint(mask_size, height - mask_size)
                    w = np.random.randint(mask_size, width - mask_size)
                    new_images[i, h-mask_size:h+mask_size, w-mask_size:w+mask_size, :] = 0
                else:
                    raise ValueError('Mask size is too large for the image input!')
            elif mask_type == 'circle':
                if height - 2*mask_size > 0 and width - 2*mask_size > 0:
                    h = np.random.randint(mask_size, height - mask_size)
                    w = np.random.randint(mask_size, width - mask_size)
                    y, x = np.ogrid[-h:height-h, -w:width-w]
                    mask = x*x + y*y <= mask_size*mask_size
                    new_images[i][mask, :] = 0
                else:
                    raise ValueError('Mask size is too large for the image input!')
            else:
                raise ValueError('Invalid mask_type, options are "square" or "circle".')

    #Reshape output from (num_images, height, width, 1) to (num_images, height, width)
    new_images = new_images.reshape(num_images, height, width)
    if num_images == 1:
        return new_images[0]
    else:
        return new_images

def image_blending(
    images, 
    num_augmentations=1, 
    blend_ratio=0.5, 
    blending_func='mean', 
    normalize_blend=True,
    num_images_to_blend=5, 
    seed=None
    ):
    """
    Blend multiple single-band images to create augmented samples.

    Parameters
    ----------
    images : ndarray
        Input as a stack (N×H×W) or a single image (H×W); values are linearly or elementwise combined to form new images. Default is required.
    num_augmentations : int, optional
        Number of blended images to generate; must be a positive integer. Default is 1.
    blend_ratio : float, optional
        Mixing weight for 'mean' blending where output = (1−blend_ratio)·A + blend_ratio·B; constrained to [0, 1]. Default is 0.5.
    blending_func : {'mean','max','min','random'}, optional
        Rule used to combine images where 'random' picks one of the other modes per blend. Default is 'mean'.
    normalize_blend : bool, optional
        If True, divide each blended image by the number of images combined to keep overall intensity comparable. Default is True.
    num_images_to_blend : int, optional
        Maximum number of distinct images sampled (uniformly without replacement) to combine per augmentation; must be ≤ N. Default is 5.
    seed : int or None, optional
        Random seed for reproducible sampling and mode selection when applicable. Default is None.

    Returns
    -------
    ndarray
        Blended images with shape (num_augmentations, H, W).
    """

    #assert images.ndim != 3, "Input images must have dimensions (num_images, height, width)"
    assert isinstance(num_augmentations, int) and num_augmentations > 0, "num_augmentations must be a positive integer"
    assert 0 <= blend_ratio <= 1, "blend_ratio must be between 0 and 1"
    assert isinstance(num_images_to_blend, int) and num_images_to_blend > 0 and num_images_to_blend <= len(images), "num_images_to_blend must be a positive integer less than or equal to the number of input images"
    
    #Define blending function
    if blending_func == 'mean':
        blend_func = lambda x, y: (1 - blend_ratio) * x + blend_ratio * y
    elif blending_func == 'max':
        blend_func = lambda x, y: np.maximum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype)
    elif blending_func == 'min':
        blend_func = lambda x, y: np.minimum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype)
    elif blending_func == 'random':
        random_func = random.choice(['mean', 'max', 'min'])
        if random_func == 'mean':
            blend_func = lambda x, y: (1 - blend_ratio) * x + blend_ratio * y
        elif random_func == 'max':
            blend_func = lambda x, y: np.maximum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype)
        elif random_func == 'min':
            blend_func = lambda x, y: np.minimum(x.astype(np.float64), y.astype(np.float64)).astype(x.dtype) 
    else:
        raise ValueError(f"Blending function '{blending_func}' not recognized, options are 'mean', 'max', 'min', or 'random'.")

    if seed is not None:
        np.random.seed(seed)
    
    #Initialize output array
    if images.ndim == 3:
        num_images, height, width = images.shape
    elif images.ndim == 2:
        height, width = images.shape 
        num_images = 1
    else:
        raise ValueError('Incorrect input shape!')

    output_images = np.zeros((num_augmentations, height, width), dtype=np.float32)
    
    #Perform image blending augmentation
    for i in range(num_augmentations):
        #Randomly select number of images to blend with, up to the num_images_to_blend
        num_images_selected = np.random.randint(2, num_images_to_blend+1)
        blend_indices = np.random.choice(num_images, size=num_images_selected, replace=False)
        blend_images = images[blend_indices]
        #Apply image blending
        blended_image = blend_images[0]
        for j in range(1, num_images_selected):
            blended_image = blend_func(blended_image, blend_images[j])
        output_images[i, :, :] += blended_image.astype(np.float32)

        if normalize_blend: #Normalize the blended images to avoid overlapping extreme pixels
            output_images[i, :, :] /= num_images_selected

    return output_images

def resize(data, size=50):
    """
    Center-crop square images to a fixed size.

    Parameters
    ----------
    data : ndarray
        Input as a single image (H×W), a stack (N×H×W), or a multi-channel stack (N×H×W×C with C ≤ 3); samples must be square. Default is required.
    size : int or None, optional
        Target side length in pixels for the center crop; if None, return the input unchanged. Default is 50.

    Returns
    -------
    ndarray
        Cropped image(s) with shape (H'×W'), (N×H'×W'), or (N×H'×W'×C), where H'=W'=size.

    Raises
    ------
    ValueError
        If input is 1D, non-square, has more than 3 channels, or has an unsupported shape.

    Notes
    -----
    - If the current side length already equals `size`, the input is returned unchanged.
    - For a single sample with channels (H×W×C), reshape to (1×H×W×C) before calling.
    """

    if size is None:
        return data 

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
        #print("No resizing necessary, image shape is already in desired size, returning original data...")
        return data 

    if len(data.shape) == 2:
        resized_data = crop_image(np.array(np.expand_dims(data, axis=-1))[:, :, 0], int(width/2.), int(height/2.), size)
        return resized_data
    else:
        resized_images = [] 
        filter1, filter2, filter3 = [], [], []
        for i in np.arange(0, len(data)):
            if len(data[i].shape) == 2:
                resized_images.append(crop_image(np.array(np.expand_dims(data[i], axis=-1))[:, :, 0], int(width/2.), int(height/2.), size))
            elif len(data[i].shape) == 3:
                if data[i].shape[-1] >= 1:
                    filter1.append(crop_image(data[i][:, :, 0], int(width/2.), int(height/2.), size))
                if data[i].shape[-1] >= 2:
                    filter2.append(crop_image(data[i][:, :, 1], int(width/2.), int(height/2.), size))
                if data[i].shape[-1] == 3:
                    filter3.append(crop_image(data[i][:, :, 2], int(width/2.), int(height/2.), size))    
                if data[i].shape[-1] > 3:
                    raise ValueError('A maximum of 3 filters is currently supported!')            
            else:
                raise ValueError('Invalid data input size, the images must be shaped as follows (# of samples, width, height, filters)')

        if len(filter1) != 0:
            for j in range(len(filter1)):
                if data[i].shape[-1] == 1:
                    resized_images.append(filter1[j])
                elif data[i].shape[-1] == 2:
                    resized_images.append(concat_channels(filter1[j], filter2[j]))
                elif data[i].shape[-1] == 3:
                    resized_images.append(concat_channels(filter1[j], filter2[j], filter3[j]))
                
    resized_data = np.array(resized_images)

    return resized_data


def random_skew(
    image: np.ndarray,
    max_angle: float = 15.0,
    intensity: float = 0.1,
    seed: Optional[int] = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    ) -> np.ndarray:
    """
    Apply a random 2‑D shear (“skew”) to an image without using OpenCV.

    Parameters
    ----------
    image : np.ndarray
        2‑D array representing the input image.
    max_angle : float, optional
        Maximum absolute shear angle (degrees) sampled independently
        for *x* and *y* directions.  Default is 15°.
    intensity : float, optional
        Additional multiplicative control on the magnitude of the shear. 
    seed : int or None, optional
        Seed for the RNG; set for reproducibility.
    order : int, optional
        Interpolation order passed to ``scipy.ndimage.affine_transform``. Defaults to 1.
    mode : {'constant', 'nearest', 'mirror', 'wrap'}, optional
        How values outside the input are filled.  Passed directly to
        ``affine_transform``.  Default is ``'constant'``.
    cval : float, optional
        Constant value used when ``mode='constant'``.  Default=0.0.

    Returns
    -------
    np.ndarray
        Skewed image with the same shape and dtype as *image*.
    """
    if image.ndim != 2:
        raise ValueError("`image` must be a 2‑D array.")

    if not (0.0 <= intensity <= 1.0):
        raise ValueError("`intensity` must lie in the interval [0,1].")

    rng = np.random.default_rng(seed)

    # Random shear angles in radians, scaled by intensity
    theta_x = np.deg2rad(rng.uniform(-max_angle, max_angle)) * intensity
    theta_y = np.deg2rad(rng.uniform(-max_angle, max_angle)) * intensity

    shear_x = np.tan(theta_x)
    shear_y = np.tan(theta_y)

    # Forward shear matrix (output → input coordinates)
    M = np.array([[1.0, shear_x],
                  [shear_y, 1.0]], dtype=float)

    # Invert because ndimage wants the mapping *from* output pixels *to* input
    M_inv = np.linalg.inv(M)

    # Center the transform so the image skews around its midpoint
    center = (np.array(image.shape[::-1]) - 1) / 2.0  # (x, y)
    offset = center - M_inv @ center

    skewed = affine_transform(
        image,
        matrix=M_inv,
        offset=offset,
        order=order,
        mode=mode,
        cval=cval,
        output_shape=image.shape,
        prefilter=(order > 1),
    )

    return skewed.astype(image.dtype)

def random_zoom(images, zoom_min=0.9, zoom_max=1.1, seed=None):
    """
    Randomly apply isotropic zoom to 2D or 3D image arrays.

    Parameters
    ----------
    images : ndarray
        Input image(s) as (H×W) or (N×H×W); 2D input is treated as a single image. Default is required.
    zoom_min : float, optional
        Lower bound on the random zoom factor; values < 1.0 shrink and > 1.0 enlarge. Default is 0.9.
    zoom_max : float, optional
        Upper bound on the random zoom factor; must be ≥ `zoom_min`. Default is 1.1.
    seed : int or None, optional
        Random seed for reproducibility; if None, use global RNG state. Default is None.

    Returns
    -------
    ndarray
        Zoomed image(s), shape matches input dimensionality: (H×W) or (N×H×W).

    Raises
    ------
    ValueError
        If `images` is not 2D or 3D, or if `zoom_max` < `zoom_min`.

    Notes
    -----
    - Nearest-neighbor interpolation is used (order=0) and edges are filled with nearest values.
    """

    if seed is not None:
        np.random.seed(seed)

    if images.ndim == 2:
        images = np.expand_dims(images, axis=0)

    zoom_factor = np.random.uniform(zoom_min, zoom_max)
    zoomed_images = zoom(images, zoom=(1.0, zoom_factor, zoom_factor), mode='nearest', order=0, prefilter=True)

    if zoomed_images.shape[0] == 1:
        zoomed_images = np.squeeze(zoomed_images, axis=0)

    return zoomed_images

def plot(data, cmap='gray', title=''):
    """
    Plot a 2D image (or stacked-channel image) with robust contrast limits.

    Parameters
    ----------
    data : ndarray
        2D array for a single-channel image, or 3D array with stacked channels (e.g., H×W×C where C∈{1,3}). Default is required.
    cmap : str, optional
        Matplotlib colormap name applied for 2D input; ignored for true-color (H×W×3). Default is 'gray'.
    title : str, optional
        Text displayed above the image; empty string shows no title. Default is ''.

    Returns
    -------
    AxesImage
        Handle from `matplotlib.pyplot.imshow`.

    Notes
    -----
    Contrast limits are computed over finite pixels using a median/MAD-like scale:
    vmin = median − 3×MAD and vmax = median + 10×MAD.
    """

    index = np.where(np.isfinite(data))
    std = np.median(np.abs(data[index]-np.median(data[index])))
    vmin = np.median(data[index]) - 3*std
    vmax = np.median(data[index]) + 10*std
    
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.show()


    