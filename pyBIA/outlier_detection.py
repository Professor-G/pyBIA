# -*- coding: utf-8 -*-
"""
Created on Wed Aug 2 06:11:11 2023

@author: danielgodinez
"""
from pathlib import Path
from progress import bar

import joblib
import numpy as np
from scipy.stats import skew, kurtosis
from skimage import exposure
from skimage.filters import gabor
from skimage.color import rgb2gray
from skimage.feature import hog, local_binary_pattern

from sklearn.ensemble import IsolationForest
from pyBIA.data_processing import process_class, concat_channels

def hog_feature_extraction(images, return_image=False, max_pool=False):
    """
    Extracts Histogram of Oriented Gradient (HOG) features from a list of images.

    This function computes HOG features from the given images. It iterates over each image and each channel of the image,
    computes the HOG features, and concatenates them. 

    If return_image is set to True, this will also return the computed HOG images for visualization.

    Args:
        images (list of numpy.ndarray): A list of input images, where each image is represented as a numpy array.
        return_image (bool, optional): Whether to return the computed HOG images along with the features. Defaults to False.
        max_pool (bool, optional): Whether to apply max pooling to HOG features within each channel. Defaults to False.

    Returns:
        Array of HOG features. 
    """

    hog_features, hog_images = [], []
    kk=0
    for image in images:
        kk+=1; print(f"{kk} out of {len(images)}")
        fd_per_channel, hog_image_per_channel = [], []
        for channel in range(image.shape[2]):   # iterate over each channel
            fd, hog_image = hog(image[:, :, channel], visualize=True)
            fd_per_channel.append(np.max(fd)) if max_pool else fd_per_channel.append(fd)
            if return_image:
                hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                hog_image_per_channel.append(hog_image_rescaled)
        if max_pool:
            hog_features.append(fd_per_channel)
        else:
            hog_features.append(np.concatenate(fd_per_channel))  # concatenate features from all channels
        if return_image:
            hog_images.append(np.stack(hog_image_per_channel, axis=-1))  # stack images from all channels
    if return_image is False:
        return np.array(hog_features)
    else:
        return np.array(hog_features), np.array(hog_images)


def extract_lbp_features(images, radius=1, n_points=8):
    """
    Extract Local Binary Pattern (LBP) features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).
        radius (int, optional): Radius for the LBP computation. Default is 1.
        n_points (int, optional): Number of points to sample around each pixel for LBP. Default is 8.

    Returns:
        Array of LPB features.
    """

    lbp_features = []
    for image in images:
        lbp_images = []
        for filter_idx in range(image.shape[-1]):
            lbp_image = local_binary_pattern(image[:, :, filter_idx], n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalize histogram
            lbp_images.append(hist)
        lbp_features.append(np.concatenate(lbp_images))
    return np.array(lbp_features)

def extract_statistical_features(images):
    """
    Extract statistical features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).

    Returns:
        Extracted statistical features for each image in the input array. Shape will be (num_images, num_filters * 4), where 4 corresponds to mean, std, skewness, and kurtosis.
    """

    num_filters = images.shape[-1]
    features = []
    for image in images:
        stats = []
        for filter_idx in range(num_filters):
            channel = image[:, :, filter_idx]
            stats.extend([np.mean(channel), np.std(channel), skew(channel.ravel()), kurtosis(channel.ravel())])
        features.append(stats)
    return np.array(features)

def extract_gabor_features(images, frequencies=[0.1, 0.3, 0.5], num_orientations=4, max_pool=True):
    """
    Extract Gabor filter features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).
        frequencies (list, optional): List of frequencies for Gabor filters. Default is [0.1, 0.3, 0.5].
        num_orientations (int, optional): Number of orientations (theta values) for Gabor filters. Default is 4.
        max_pool (bool, optional): Whether to max pool each frequency-orientation pair so as to avoid returning a large array,
            at the cost of information loss.
    
    Returns:
        Array of Gabor filter features. Shape will be (num_images, num_filters * num_orientations * len(frequencies)).
    """

    gray_images = np.mean(images, axis=-1)
    gabor_responses = []
    for image in gray_images:
        gabor_features = []
        for frequency in frequencies:
            for theta in range(num_orientations):
                gabor_response = np.abs(gabor(image, frequency, theta=theta)[0])
                if max_pool:
                    gabor_features.append(np.max(gabor_response))  # Apply max pooling to each frequency-orientation pair
                else:
                    gabor_features.extend(gabor_response.ravel())
        gabor_responses.append(gabor_features)
    return np.array(gabor_responses)

def extract_color_moments(images):
    """
    Extract color moment features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).

    Returns:
        Array ofcolor moments. Shape will be (num_images, num_filters * 4), where 4 corresponds to mean, std, skewness, and kurtosis.
    """

    moments_features = []
    for image in images:
        color_features = []
        for filter_idx in range(image.shape[-1]):
            channel = image[:, :, filter_idx]
            moments = [np.mean(channel), np.std(channel), skew(channel.ravel()), kurtosis(channel.ravel())]
            color_features.extend(moments)
        moments_features.append(color_features)
    return np.array(moments_features)

def extract_histogram_features(images, bins=256):
    """
    Extract histogram features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).
        bins (int, optional): Number of bins for the histogram. Default is 256.

    Returns:
        Array of histogram features. Shape will be (num_images, num_filters * bins).
    """

    histogram_features = []
    for image in images:
        hist_features = []
        for filter_idx in range(image.shape[-1]):
            hist, _ = np.histogram(image[:, :, filter_idx].ravel(), bins=bins, range=(0, 1))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalize histogram
            hist_features.extend(hist)
        histogram_features.append(hist_features)
    return np.array(histogram_features)

def extract_gradient_features(images, bins=256):
    """
    Extract gradient magnitude histogram features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).
        bins (int, optional): Number of bins for the gradient magnitude histogram. Default is 256.

    Returns:
        Array of gradient magnitude histogram features. Shape will be (num_images, num_filters * bins).
    """

    gradient_features = []
    for image in images:
        grad_features = []
        for filter_idx in range(image.shape[-1]):
            gradient_magnitude = np.sqrt(np.square(np.gradient(image[:, :, filter_idx], axis=0)) +
                                        np.square(np.gradient(image[:, :, filter_idx], axis=1)))
            hist, _ = np.histogram(gradient_magnitude.ravel(), bins=bins)
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalize histogram
            grad_features.extend(hist)
        gradient_features.append(grad_features)
    return np.array(gradient_features)

def extract_contrast_features(images, bins=256):
    """
    Extract contrast-enhanced histogram features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).
        bins (int, optional): Number of bins for the contrast-enhanced histogram. Default is 256.

    Returns:
        Array of contrast-enhanced histogram features. Shape will be (num_images, num_filters * bins).
    """

    contrast_features = []
    for image in images:
        cont_features = []
        for filter_idx in range(image.shape[-1]):
            contrast_image = exposure.equalize_adapthist(image[:, :, filter_idx])
            hist, _ = np.histogram(contrast_image.ravel(), bins=bins, range=(0, 1))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalize histogram
            cont_features.extend(hist)
        contrast_features.append(cont_features)
    return np.array(contrast_features)

def extract_intensity_features(images, bins=256):
    """
    Extract intensity histogram features from a collection of images.

    Parameters:
        images (ndarray): Array of images with shape (num_images, width, height, num_filters).
        bins (int, optional): Number of bins for the intensity histogram. Default is 256.

    Returns:
        Array of intensity histogram features. Shape will be (num_images, num_filters * bins).
    """

    intensity_features = []
    for image in images:
        int_features = []
        for filter_idx in range(image.shape[-1]):
            hist, _ = np.histogram(image[:, :, filter_idx].ravel(), bins=bins, range=(0, 1))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)  # Normalize histogram
            int_features.extend(hist)
        intensity_features.append(int_features)
    return np.array(intensity_features)



#####

"""

from pyBIA.data_augmentation import resize 

image_sizes = np.arange(50, 251, 5)

for image_size in image_sizes:
    confirmed_diffuse = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/final_analysis/images/confirmed_diffuse/confirmed_diffuse.npy')
    priority_diffuse = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/final_analysis/images/priority_diffuse/priority_diffuse.npy')
    other_diffuse =  np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/final_analysis/images/other_diffuse/other_diffuse.npy')
    #other_diffuse_names = np.loadtxt('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/images/OTHER_XGB_FEATS_LARGER/other_diffuse/other_diffuse_names.txt', dtype=str)
    all_diffuse = np.vstack((confirmed_diffuse, priority_diffuse, other_diffuse))
    all_diffuse = resize(all_diffuse, image_size)
    all_diffuse_normalized = process_class(all_diffuse, normalize=True, min_pixel=0, max_pixel=[10,10], img_num_channels=2)
    f1 = hog_feature_extraction(all_diffuse_normalized)
    f2 = extract_lbp_features(all_diffuse_normalized)
    f3 = extract_statistical_features(all_diffuse_normalized)
    f4 = extract_gabor_features(all_diffuse_normalized)
    f5 = extract_color_moments(all_diffuse_normalized)
    f6 = extract_histogram_features(all_diffuse_normalized)
    f7 = extract_gradient_features(all_diffuse_normalized)
    f8 = extract_contrast_features(all_diffuse_normalized)
    f9 = extract_intensity_features(all_diffuse_normalized)
    np.savez('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/final_analysis/all_diffuse_hog_features_'+str(image_size)+'pix.npz', f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, f6=f6, f7=f7, f8=f8, f9=f9)

# Extract the HOG features for the negative class (randomly selected outliers)

for image_size in image_sizes:
    other_bw = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/images/OTHER_XGB_FEATS_LARGER/xgb_output_images_bw_LARGE.npy')
    other_r = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/images/OTHER_XGB_FEATS_LARGER/xgb_output_images_R_LARGE.npy')
    #
    outlier_indices = np.loadtxt('/Users/daniel/Desktop/outliers_indices_866.txt')
    outlier_indices = [int(i) for i in outlier_indices]
    outliers = concat_channels(other_bw[outlier_indices], other_r[outlier_indices])
    outliers = resize(outliers, image_size)
    outliers = process_class(outliers, normalize=True, min_pixel=0, max_pixel=[10,10], img_num_channels=2)
    f1 = hog_feature_extraction(outliers)
    f2 = extract_lbp_features(outliers)
    f3 = extract_statistical_features(outliers)
    f4 = extract_gabor_features(outliers)
    f5 = extract_color_moments(outliers)
    f6 = extract_histogram_features(outliers)
    f7 = extract_gradient_features(outliers)
    f8 = extract_contrast_features(outliers)
    f9 = extract_intensity_features(outliers)
    np.savez('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/outliers_features_'+str(image_size)+'pix.npz', f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, f6=f6, f7=f7, f8=f8, f9=f9)
    #
    non_outlier_indices = np.loadtxt('/Users/daniel/Desktop/250_indices.txt')
    non_outlier_indices = [int(i) for i in non_outlier_indices]
    non_outliers = concat_channels(other_bw[non_outlier_indices], other_r[non_outlier_indices])
    non_outliers = resize(non_outliers, image_size)
    non_outliers = process_class(non_outliers, normalize=True, min_pixel=0, max_pixel=[10,10], img_num_channels=2)
    f1 = hog_feature_extraction(non_outliers)
    f2 = extract_lbp_features(non_outliers)
    f3 = extract_statistical_features(non_outliers)
    f4 = extract_gabor_features(non_outliers)
    f5 = extract_color_moments(non_outliers)
    f6 = extract_histogram_features(non_outliers)
    f7 = extract_gradient_features(non_outliers)
    f8 = extract_contrast_features(non_outliers)
    f9 = extract_intensity_features(non_outliers)
    np.savez('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/non_outliers_features_'+str(image_size)+'pix.npz', f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, f6=f6, f7=f7, f8=f8, f9=f9)
    #


"""

from sklearn.ensemble import IsolationForest
import numpy as np  
import joblib 


image_sizes = np.arange(50, 251, 5)

#for image_size in image_sizes:

image_size=225

diffuse_feats = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/outlier_detection/diffuse/all_diffuse_features_'+str(image_size)+'pix.npz')
f1_diffuse = diffuse_feats['f1'] #hog_feature_extraction
f2_diffuse = diffuse_feats['f2'] #extract_lbp_features
f3_diffuse = diffuse_feats['f3'] #extract_statistical_features
f4_diffuse = diffuse_feats['f4'] #extract_gabor_features
f5_diffuse = diffuse_feats['f5'] #extract_color_moments
f6_diffuse = diffuse_feats['f6'] #extract_histogram_features
f7_diffuse = diffuse_feats['f7'] #extract_gradient_features
f8_diffuse = diffuse_feats['f8'] #extract_contrast_features
f9_diffuse = diffuse_feats['f9'] #extract_intensity_features


outlier_feats = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/outlier_detection/outliers/outliers_features_'+str(image_size)+'pix.npz')
f1_outlier = outlier_feats['f1']
f2_outlier = outlier_feats['f2']
f3_outlier = outlier_feats['f3']
f4_outlier = outlier_feats['f4']
f5_outlier = outlier_feats['f5']
f6_outlier = outlier_feats['f6']
f7_outlier = outlier_feats['f7']
f8_outlier = outlier_feats['f8']
f9_outlier = outlier_feats['f9']


non_outlier_feats = np.load('/Users/daniel/Desktop/Folders/Lyalpha/pyBIA_Paper_1/outlier_detection/non_outliers/non_outliers_features_'+str(image_size)+'pix.npz')
f1_non_outlier = non_outlier_feats['f1']
f2_non_outlier = non_outlier_feats['f2']
f3_non_outlier = non_outlier_feats['f3']
f4_non_outlier = non_outlier_feats['f4']
f5_non_outlier = non_outlier_feats['f5']
f6_non_outlier = non_outlier_feats['f6']
f7_non_outlier = non_outlier_feats['f7']
f8_non_outlier = non_outlier_feats['f8']
f9_non_outlier = non_outlier_feats['f9']

model = IsolationForest()
model.fit(f4_diffuse)

y_pred = model.predict(f4_diffuse)
len(np.where(y_pred==1)[0]) / len(y_pred)

y_pred = model.predict(f4_outlier)
len(np.where(y_pred==-1)[0]) / len(y_pred)

y_pred = model.predict(f4_non_outlier)
len(np.where(y_pred==1)[0]) / len(y_pred)

######


feats_diffuse = np.c_[f1_diffuse, f2_diffuse, f4_diffuse, f7_diffuse]
feats_outlier = np.c_[f1_outlier, f2_outlier, f4_outlier, f7_outlier]
feats_non_outlier = np.c_[f1_non_outlier, f2_non_outlier, f4_non_outlier, f7_non_outlier]

model = IsolationForest()
model.fit(feats_diffuse)

y_pred = model.predict(feats_diffuse)
len(np.where(y_pred==1)[0]) / len(y_pred)

y_pred = model.predict(feats_outlier)
len(np.where(y_pred==-1)[0]) / len(y_pred)

y_pred = model.predict(feats_non_outlier)
len(np.where(y_pred==1)[0]) / len(y_pred)





