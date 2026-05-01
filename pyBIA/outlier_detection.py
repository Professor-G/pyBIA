# -*- coding: utf-8 -*-
"""
Created on Wed Aug 2 06:11:11 2023

@author: danielgodinez
"""
import numpy as np
from skimage import exposure
from skimage.feature import hog, local_binary_pattern
from scipy.stats import skew, kurtosis, entropy
from typing import List, Tuple, Union, Optional
from pathlib import Path
import joblib 
import os 

import pywt
from sklearn.ensemble import IsolationForest
from pyBIA.data_processing import process_class
from pyBIA.optimization import impute_missing_values

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler 


class Classifier:
    """
    Build and apply an ensemble outlier-detection classifier on image cutouts.

    The classifier workflow supports optional min–max normalization, feature
    extraction (HOG, LBP, FFT, Wavelet, or simple statistics), optional
    imputation of missing values, and model fitting using an Isolation Forest
    (`clf='iforest'`). 

    If multiple feature sets are provided, an independent pipeline (imputer, 
    scaler, PCA, and model) is trained for each feature set. Predictions are 
    made by either averaging the anomaly scores across all independent models
    or by selecting only the most anomolous score across all models.

    Parameters
    ----------
    data : ndarray or None, optional
        Image tensor with shape (N, H, W, C).
    normalize : bool, optional
        If True, min–max normalize each image/channel before feature extraction.
    min_pixel : float, optional
        Lower bound for min–max normalization.
    max_pixel : float, optional
        Upper bound for min–max normalization.
    img_num_channels : int, optional
        Number of channels in the input tensor.
    feat_set : str or list/tuple of str, optional
        Feature family (or families) to compute for training. 
        Options: 'hog','lbp','fft','wavelet','stats'. 
    clf : {'iforest'}, optional
        Classifier to train. Currently only Isolation Forest is supported.
    impute : bool, optional
        If True, impute missing feature values before fitting/predicting.
    imp_method : {'knn','mean','median','mode','constant'}, optional
        Imputation strategy used by `impute_missing_values`.
    scale_features : bool, optional
        If True, scales extracted features.
    scaler_type : {'robust', 'standard'}, optional
        Type of scaler to use. Default is 'robust'.
    apply_pca : bool, optional
        If True, performs Principal Component Analysis on the extracted features.
    pca_components : int or float, optional
        Number of components to keep.
    SEED_NO : int, optional
        Random seed used for model initialization. Default is 1909.

    Attributes
    ----------
    models : dict
        Dictionary of trained models keyed by feature name.
    imputers : dict
        Dictionary of fitted imputers keyed by feature name.
    scalers : dict
        Dictionary of fitted scalers keyed by feature name.
    pcas : dict
        Dictionary of fitted PCA models keyed by feature name.
    """

    def __init__(self, 
        data=None,
        normalize=False,
        min_pixel=0,
        max_pixel=10,
        img_num_channels=1,
        feat_set='hog',
        clf='iforest', 
        impute=True, 
        imp_method='knn', 
        scale_features=False,
        scaler_type='robust',
        apply_pca=False,
        pca_components=None,
        SEED_NO=1909
        ):

        self.data = data
        self.normalize = normalize
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.img_num_channels = img_num_channels
        self.clf = clf
        self.impute = impute
        self.imp_method = imp_method
        self.scale_features = scale_features
        self.scaler_type = scaler_type
        self.apply_pca = apply_pca
        self.pca_components = pca_components
        self.SEED_NO = SEED_NO

        # Dicts to hold independent pipelines for each feature set
        self.models = {}
        self.imputers = {}
        self.scalers = {}
        self.pcas = {}

        if isinstance(feat_set, str):
            self.feat_set = [feat_set]
        elif isinstance(feat_set, (list, tuple)):
            self.feat_set = list(feat_set)
        else:
            raise ValueError('The `feat_set` must be a string or a list/tuple of strings.')

        valid_feats = {'hog', 'lbp', 'fft', 'wavelet', 'stats'}
        for feat in self.feat_set:
            if feat not in valid_feats:
                raise ValueError(f'The `feat_set` input "{feat}" is invalid, options are: {valid_feats}')

        if scaler_type not in ('robust', 'standard'):
            raise ValueError('The `scaler_type` input is invalid, options are: "robust", "standard"')

    def _extract_single_feature(self, data, feat: str) -> np.ndarray:
        """
        Extract a single feature matrix.

        Parameters
        ----------
        data : ndarray
            Input image tensor of shape (N, H, W, C).
        feat : {'hog', 'lbp', 'fft', 'wavelet', 'stats'}
            Feature extraction method to apply:
            - 'hog' : Histogram of Oriented Gradients
            - 'lbp' : Local Binary Patterns
            - 'fft' : Fourier-based energy features
            - 'wavelet' : Wavelet energy features
            - 'stats' : Simple statistical features

        Returns
        -------
        f_data : ndarray
            Extracted feature matrix of shape (N, D), where D depends on the
            selected feature type.
        """

        if feat == 'hog':
            f_data = hog_feature_extraction(data)
        elif feat == 'lbp':
            f_data = lbp_feature_extraction(data)
        elif feat == 'wavelet':
            f_data = wavelet_energy_feature_extraction(data)
        elif feat == 'fft':
            f_data = fft_energy_feature_extraction(data)
        elif feat == 'stats':
            f_data = statistical_feature_extraction(data)
        
        if len(f_data.shape) == 1:
            f_data = f_data.reshape(1, -1)
            
        return f_data

    def create(self, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0):
        """
        Initialize, featurize, optionally impute, and fit the classifier.
        This method instantiates the model, optionally normalizes the data using the min-max bounds,
        extracts the features, replaces inf with NaNs, and then optionally imputes missing values.
        The model is then fitted on the resulting feature matrix. The optional arguments are
        iForest hyperparameters, which by default are the scikit-learn defaults.

        Parameters
        ----------
        n_estimators : int
            Number of trees to fit. Defaults to 100.
        max_samples : 'auto' or int
            The number of training instances to use to train the model. Defaults to 'auto'.
        contamination : float
            The expected ratio of outliers present in the training data. Sets what the anomaly 
            score threshold should be. Defaults to 'auto'.
        max_features : int or float
            The number (or proportion if float) of training features to draw from the feature matrix when training the model.
            Defaults to 1.0

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an unsupported `clf` is requested.
        ValueError
            If `impute=False` and the feature matrix contains NaNs or infs.
        """
    
        if self.clf != 'iforest':
            raise ValueError('Only IsolationForest is currently supported! Set `clf`="iforest" and run again.')
        
        if self.normalize:
            self.data = process_class(
                self.data, 
                normalize=self.normalize, 
                min_pixel=self.min_pixel, 
                max_pixel=[self.max_pixel]*self.img_num_channels, 
                img_num_channels=self.img_num_channels
                )

        # Train an independent pipeline for each feature set
        for feat in self.feat_set:
            print(f"Training pipeline for feature: {feat}")
            data_x = self._extract_single_feature(self.data, feat)
            data_x[np.isinf(data_x)] = np.nan

            if self.impute is False:
                if np.any(np.isfinite(data_x) == False):
                    raise ValueError(f'Feature array for {feat} contains nan values but `impute` is False!')
            else:
                data_x, imputer = impute_missing_values(data_x, strategy=self.imp_method)
                self.imputers[feat] = imputer
            
            if self.scale_features:
                scaler = RobustScaler() if self.scaler_type == 'robust' else StandardScaler()
                data_x = scaler.fit_transform(data_x)
                self.scalers[feat] = scaler

            if self.apply_pca:
                pca = PCA(n_components=self.pca_components, random_state=self.SEED_NO)
                data_x = pca.fit_transform(data_x)
                self.pcas[feat] = pca

            # Fit independent Isolation Forest
            model = IsolationForest(
                n_estimators=n_estimators, 
                max_samples=max_samples, 
                contamination=contamination, 
                max_features=max_features, 
                random_state=self.SEED_NO
                )

            model.fit(data_x)
            self.models[feat] = model

        print(f"Successfully trained {len(self.models)} independent models for ensemble.")
        return
        
    def save(self, dirname=None, path=None, overwrite=False):
        """
        Save the trained model (and imputer/scaler/pca if present) to disk.

        Creates a directory `pyBIA_outlier_model` under `path[/dirname]/` and
        writes the IsolationForest model and the fitted imputer/scaler/pca, if applicable.

        Parameters
        ----------
        dirname : str or None, optional
            Optional subdirectory to create inside `path`. Must not already exist.
        path : str or None, optional
            Base directory where the model folder will be saved. If None, uses the
            user's home directory.
        overwrite : bool, optional
            If True and `pyBIA_outlier_model` exists, delete its contents and
            recreate it. If False, raise if the folder exists. Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no artifacts are available to save (e.g., model not created).
        ValueError
            If attempting to create an existing directory without `overwrite=True`.
        """

        if not self.models:
            raise ValueError('The models have not been created! Run the create() method first.')

        path = str(Path.home()) if path is None else path 
        path = path + '/' if path[-1] != '/' else path 
        
        if dirname is not None:
            dirname = dirname + '/' if dirname[-1] != '/' else dirname
            path = path + dirname
            try:
                os.makedirs(path)
            except FileExistsError:
                raise ValueError('The dirname folder already exists!')

        try:
            os.mkdir(path + 'pyBIA_outlier_model')
        except FileExistsError:
            if overwrite:
                import shutil
                shutil.rmtree(path + 'pyBIA_outlier_model')
                os.mkdir(path + 'pyBIA_outlier_model')
            else:
                raise ValueError('Directory already exists! Overwrite=False.')
        
        path += 'pyBIA_outlier_model/'

        # Save artifacts per feature
        for feat in self.feat_set:
            if feat in self.models:
                joblib.dump(self.models[feat], path + f'Model_{feat}')
            if feat in self.imputers:
                joblib.dump(self.imputers[feat], path + f'Imputer_{feat}')
            if feat in self.scalers:
                joblib.dump(self.scalers[feat], path + f'Scaler_{feat}')
            if feat in self.pcas:
                joblib.dump(self.pcas[feat], path + f'PCA_{feat}')

        print(f'Files saved in: {path}')
        self.path = path
        return 

    def load(self, path=None):
        """ 
        Load a saved model (and imputer/scaler/pca if present) from disk.

        Looks for a folder named `pyBIA_outlier_model` under `path` (or the user’s
        home directory if `path` is None) and attempts to load `Model` and `Imputer`
        artifacts into `self.model` and `self.imputer`.

        Parameters
        ----------
        path : str or None, optional
            Base directory containing `pyBIA_outlier_model/`. If None, uses the user's home directory.

        Returns
        -------
        None
        """

        path = str(Path.home()) if path is None else path 
        path = path+'/' if path[-1] != '/' else path 
        path += 'pyBIA_outlier_model/'

        self.models = {}
        self.imputers = {}
        self.scalers = {}
        self.pcas = {}

        loaded_components = []

        for feat in self.feat_set:
            if os.path.exists(path + f'Model_{feat}'):
                self.models[feat] = joblib.load(path + f'Model_{feat}')
                loaded_components.append(f'Model_{feat}')
            
            if os.path.exists(path + f'Imputer_{feat}'):
                self.imputers[feat] = joblib.load(path + f'Imputer_{feat}')
                
            if os.path.exists(path + f'Scaler_{feat}'):
                self.scalers[feat] = joblib.load(path + f'Scaler_{feat}')
                
            if os.path.exists(path + f'PCA_{feat}'):
                self.pcas[feat] = joblib.load(path + f'PCA_{feat}')

        print(f'Successfully loaded pipelines for: {list(self.models.keys())}')
        self.path = path
        return

    def predict(self, data, ensemble_method='strict'):
        """
        Predict outlier/inlier labels via ensemble aggregation.

        Parameters
        ----------
        data : ndarray
            Image tensor with shape (N, H, W, C).
        ensemble_method : {'average', 'strict'}, optional
            How to combine the scores from the independent models. 
            'average' computes the mean of the scores.
            'strict' takes the minimum score, therefore if any model 
            flags the sample as an anomaly, it will be marked as an anomaly.
            Default is 'strict'.
        """

        if not self.models:
            raise ValueError('No models have been created or loaded! Run create() or load() first!')

        if self.normalize:
            data = process_class(
                data, 
                normalize=self.normalize, 
                min_pixel=self.min_pixel, 
                max_pixel=[self.max_pixel]*self.img_num_channels, 
                img_num_channels=self.img_num_channels
                )

        all_decision_scores = []
        all_raw_scores = []

        for feat in self.feat_set:
            if feat not in self.models:
                raise ValueError(f"Feature '{feat}' was requested but no trained model exists for it.")

            data_x = self._extract_single_feature(data, feat)

            if feat in self.imputers:
                data_x = self.imputers[feat].transform(data_x)
            elif np.any(np.isfinite(data_x) == False):
                 raise ValueError(f'data_x for {feat} contains nan but no imputer exists!')

            if feat in self.scalers:
                data_x = self.scalers[feat].transform(data_x)

            if feat in self.pcas:
                data_x = self.pcas[feat].transform(data_x)

            # Get scores from this specific model
            model = self.models[feat]
            decision_scores = model.decision_function(data_x)
            raw_scores = decision_scores + model.offset_

            all_decision_scores.append(decision_scores)
            all_raw_scores.append(raw_scores)

        # Model aggregation options
        if ensemble_method == 'average': # Average score from all models
            agg_decision_scores = np.mean(all_decision_scores, axis=0)
            agg_raw_scores = np.mean(all_raw_scores, axis=0)
        elif ensemble_method == 'strict': # In this case the lowest score represents the highest confidence of an anomaly
            agg_decision_scores = np.min(all_decision_scores, axis=0)
            agg_raw_scores = np.min(all_raw_scores, axis=0)
            
        else:
            raise ValueError("ensemble_method must be either 'average' or 'strict'")

        # If the aggregated score is < 0 set -1 (outlier), otherwise 1 (inlier)
        predictions = np.where(agg_decision_scores < 0, -1, 1) 
        
        return np.c_[predictions, agg_decision_scores, agg_raw_scores]


def hog_feature_extraction(
    images, 
    return_image=False, 
    max_pool=False
    ):
    """
    Extract Histogram of Oriented Gradients (HOG) features per channel.

    Parameters
    ----------
    images : ndarray
        Input tensor of shape (N, H, W, C), where N is the number of images,
        H×W are spatial dimensions, and C is the number of channels.
    return_image : bool, optional
        If True, also return the HOG visualization images (per channel), stacked
        along the last axis. Default is False.
    max_pool : bool, optional
        If True, apply global max pooling to each per-channel HOG feature vector
        (i.e., keep only its maximum value). The resulting feature for each image
        has shape (C,). If False, per-channel feature vectors are concatenated.
        Default is False.

    Returns
    -------
    hog_features : ndarray
        If `max_pool=False`: array of shape (N, D), where D is the sum of HOG
        feature lengths across channels (concatenated).
        If `max_pool=True`: array of shape (N, C), one scalar per channel.
    hog_images : ndarray, optional
        Returned only if `return_image=True`. Array of shape (N, H, W, C),
        containing per-channel HOG visualizations rescaled to display range.

    Raises
    ------
    ValueError
        If `images` does not have 4 dimensions (N, H, W, C).

    Notes
    -----
    - Each channel is treated as a grayscale image (`channel_axis=None`).
    - HOG parameters are the scikit-image defaults (orientations, pixels per cell, cells per block, block normalization).
    """

    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("images must have shape (N, H, W, C).")

    N, H, W, C = images.shape
    hog_features, hog_images = [], []

    for i in range(N):
        fd_per_channel, hog_image_per_channel = [], []
        for ch in range(C):
            chan = images[i, :, :, ch].astype(np.float64)

            if return_image:
                fd, hog_img = hog(
                    chan,
                    visualize=True,
                    feature_vector=True,
                    channel_axis=None,  # grayscale input
                )
                hog_image_per_channel.append(
                    exposure.rescale_intensity(hog_img, in_range="image")
                )
            else:
                fd = hog(
                    chan,
                    visualize=False,
                    feature_vector=True,
                    channel_axis=None,
                )

            fd_per_channel.append(fd.max() if max_pool else fd)

        hog_features.append(
            np.asarray(fd_per_channel) if max_pool else np.concatenate(fd_per_channel)
        )
        if return_image:
            hog_images.append(np.stack(hog_image_per_channel, axis=-1))

    hog_features = np.asarray(hog_features, dtype=np.float64)
    if return_image:
        return hog_features, np.asarray(hog_images)

    return hog_features

def wavelet_energy_feature_extraction(
    images: List[np.ndarray],
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "symmetric",
    stat: str = "sum",
    log_scale: bool = True,
    normalize: bool = False,
    eps: float = 1e-10
    ) -> np.ndarray:
    """
    Compute per-subband wavelet energies per channel and concatenate.

    Parameters
    ----------
    images : sequence of ndarray or ndarray
        Iterable of images with shape `(H, W, C)` or an array with shape
        `(N, H, W, C)`. Iteration is over the first dimension.
    wavelet : str, optional
        Wavelet name for PyWavelets. Default is 'db4'.
    level : int or None, optional
        Decomposition level `L`. If None, uses the maximum level allowed by the
        image size and wavelet filter length. Default is None.
    mode : str, optional
        Boundary extension mode passed to `pywt.wavedec2`. Default is 'symmetric'.
    stat : {'sum','mean'}, optional
        Aggregation for each subband:
        - 'sum' : sum of squares (energy)
        - 'mean' : energy per coefficient (area-normalized)
        Default is 'sum'.
    log_scale : bool, optional
        If True, apply `log(energy + eps)` to each subband value. Default is True.
    normalize : bool, optional
        If True, divide all subband values in a channel by that channel's total
        (after `stat`), for relative energies. Default is False.
    eps : float, optional
        Small constant used in log/normalization to avoid division by zero and
        `log(0)`. Default is 1e-10.

    Returns
    -------
    feats : ndarray, shape (N, C * (1 + 3L))
        Wavelet-energy feature matrix. For each image (N) and channel (C), the
        feature length is `1 + 3L` (one approximation band + three detail bands
        per level), concatenated across channels.
    """

    feats_all = []
    w = pywt.Wavelet(wavelet)

    for img in images:
        ch_feats = []
        H, W, C = img.shape
        # choose level once per image
        max_level = pywt.dwt_max_level(min(H, W), w.dec_len)
        L = max_level if level is None else int(level)
        if L < 1 or L > max_level:
            raise ValueError(f"level must be in [1, {max_level}] for size {H}x{W} and wavelet {wavelet}")

        for ch in range(C):
            chan = np.asarray(img[..., ch], dtype=np.float64)
            coeffs = pywt.wavedec2(chan, wavelet=w, mode=mode, level=L)
            # coeffs = [cA_L, (cH_L,cV_L,cD_L), ..., (cH_1,cV_1,cD_1)]

            energies = []
            sizes = []

            # Approximation energy at level L
            cA = coeffs[0]
            energies.append(np.sum(cA**2))
            sizes.append(cA.size)

            # Detail energies for levels L..1
            for (cH, cV, cD) in coeffs[1:]:
                for band in (cH, cV, cD):
                    energies.append(np.sum(band**2))
                    sizes.append(band.size)

            energies = np.asarray(energies, dtype=float)
            sizes = np.asarray(sizes, dtype=float)

            if stat == "mean":
                # area-normalize to remove bias from subband size
                energies = np.divide(energies, sizes, out=np.zeros_like(energies), where=sizes > 0)
            elif stat != "sum":
                raise ValueError("stat must be 'sum' or 'mean'.")

            if normalize:
                total = energies.sum() + eps
                energies = energies / total

            if log_scale:
                energies = np.log(energies + eps)

            ch_feats.append(energies)

        feats_all.append(np.concatenate(ch_feats))

    return np.asarray(feats_all, dtype=np.float64)

def statistical_feature_extraction(images: np.ndarray) -> np.ndarray:
    """
    Compute global statistics and simple texture descriptors per channel.

    For each image channel, the following 10 features are computed over finite
    pixels only and concatenated across channels:

        1) mean
        2) std (population, ddof=0)
        3) median
        4) median absolute deviation (MAD)
        5) 1st percentile (p01)
        6) 99th percentile (p99)
        7) min
        8) max
        9) skewness
       10) kurtosis

    Parameters
    ----------
    images : ndarray, shape (N, H, W, C)
        Image tensor (floats). Non-finite values (NaN/±inf) are ignored when
        computing per-channel statistics.

    Returns
    -------
    feats : ndarray, shape (N, C * 10)
        Per-image feature matrix, dtype float64. Features are ordered as listed
        above for channel 0, then channel 1, etc.

    Raises
    ------
    ValueError
        If `images` does not have 4 dimensions (N, H, W, C).
    """

    if images.ndim != 4:
        raise ValueError("Input must be (N, H, W, C).")

    pctl = lambda a, q: np.percentile(a, q)

    n, _, _, c = images.shape
    feats = np.empty((n, c * 10), dtype=np.float64)

    for i, img in enumerate(images):
        stats = []
        for ch in range(c):
            chan = img[..., ch]
            chan = chan[np.isfinite(chan)]
            flat = chan.ravel()

            mu = np.mean(flat)
            sigma = np.std(flat)
            med = np.median(flat)
            mad = np.median(np.abs(flat - med))
            p01 = pctl(flat, 1)
            p99 = pctl(flat, 99)
            amin = np.min(flat)
            amax = np.max(flat)
            _skew = skew(flat)
            _kurt = kurtosis(flat)

            stats.extend([mu, sigma, med, mad, p01, p99, amin, amax, _skew, _kurt])

        feats[i] = stats

    return feats

def lbp_feature_extraction(
    images, 
    P: int = 8, 
    R: int = 1
    ):
    """
    Extract Local Binary Pattern (LBP) histograms per channel and concatenate.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, C)
        Input image tensor. Each channel is treated independently.
    P : int, optional
        Number of sampling points on the LBP circle. Default is 8.
    R : int, optional
        Radius (in pixels) of the LBP circle. Default is 1.

    Returns
    -------
    feats : ndarray, shape (N, C * 2**P)
        Concatenated per-channel LBP histograms, dtype float64.
        For each image, the feature for channel 0 is first, then channel 1, etc.

    Raises
    ------
    ValueError
        If `images` does not have 4 dimensions (N, H, W, C).
    """

    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("images must have shape (N, H, W, C).")

    N, H, W, C = images.shape
    bins = np.arange(0, 2**P + 1)  # 2**P bins for 'default'

    feats = []
    for i in range(N):
        per_chan = []
        for ch in range(C):
            chan = images[i, :, :, ch].astype(np.float32)
            lbp = local_binary_pattern(chan, P=P, R=R, method="default")
            hist, _ = np.histogram(lbp.ravel(), bins=bins, density=True)
            per_chan.append(hist)
        feats.append(np.concatenate(per_chan))

    return np.asarray(feats, dtype=np.float64)

def fft_energy_feature_extraction(
    images,
    band_edges=(0.0, 0.10, 0.25, 0.50, 0.75, 1.0),
    per_band_norm=True,
    window=True,
    stat='sum',
    remove_dc=True,
    fft_norm=None
    ):
    """
    Compute 2D FFT radial-band energies per channel and concatenate.

    Parameters
    ----------
    images : ndarray, shape (N, H, W, C)
        Input cutouts. Channels are processed independently and concatenated.
    band_edges : sequence of float, optional
        Strictly increasing edges within [0, 1], defining bands
        `[edges[i], edges[i+1])`, with the last band including its upper edge.
        Default is (0.0, 0.10, 0.25, 0.50, 0.75, 1.0).
    per_band_norm : bool, optional
        If True, divide each channel’s band energies by their sum so that the
        per-channel features sum to 1. Default is True.
    window : bool, optional
        If True, apply a separable Hann window prior to FFT. Default is True.
    stat : {'sum','mean'}, optional
        Aggregation within each annulus:
        - 'sum'  : sum of power (energy)
        - 'mean' : average power per coefficient (area-normalized)
        Default is 'sum'.
    remove_dc : bool, optional
        If True, zero the DC coefficient so features emphasize texture rather
        than mean flux. Default is True.
    fft_norm : {None, 'ortho'}, optional
        Normalization passed to `numpy.fft.fft2`. Default is None.

    Returns
    -------
    feats : ndarray, shape (N, C * (len(band_edges) - 1))
        Concatenated per-channel band features (float64). If `per_band_norm=True`,
        each channel’s bands sum to 1 for a given image.

    Raises
    ------
    ValueError
        If `images` is not (N, H, W, C).
    """

    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError("images must have shape (N, H, W, C).")
    if stat not in {'sum', 'mean'}:
        raise ValueError("stat must be 'sum' or 'mean'.")

    n, h, w, c = images.shape
    edges = np.asarray(band_edges, dtype=float)
    if not (np.all(np.diff(edges) > 0) and 0.0 <= edges[0] and edges[-1] <= 1.0 + 1e-12):
        raise ValueError("band_edges must be strictly increasing within [0, 1].")

    nb = len(edges) - 1
    feats = np.zeros((n, c * nb), dtype=np.float64)
    eps = 1e-12

    # Nyquist-normalized radial grid: r = sqrt(kx^2 + ky^2) / 0.5
    ky = np.fft.fftfreq(h) # cycles/pixel in [-0.5, 0.5)
    kx = np.fft.fftfreq(w)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    R = np.sqrt(KY**2 + KX**2) / 0.5
    R = np.fft.fftshift(R)
    inside = (R <= 1.0 + 1e-12) # only coefficients within unit circle

    # Annular masks within unit circle (include edge in last band)
    masks = []
    for i in range(nb):
        lo, hi = edges[i], edges[i+1]
        if i < nb - 1:
            m = (R >= lo) & (R < hi) & inside
        else:
            m = (R >= lo) & (R <= hi + 1e-12) & inside
        masks.append(m)
    counts = np.array([m.sum() for m in masks], dtype=float)

    # Hann window
    if window:
        wy = 0.5 * (1 - np.cos(2 * np.pi * np.arange(h) / max(h - 1, 1)))
        wx = 0.5 * (1 - np.cos(2 * np.pi * np.arange(w) / max(w - 1, 1)))
        W = wy[:, None] * wx[None, :]
    else:
        W = 1.0

    cy, cx = h // 2, w // 2  # DC index after fftshift

    for i in range(n):
        row = []
        img = images[i]
        for ch in range(c):
            patch = img[..., ch].astype(np.float64) * W
            F = np.fft.fft2(patch, norm=fft_norm)
            P = np.fft.fftshift(np.abs(F)**2)

            if remove_dc:
                P[cy, cx] = 0.0

            band_vals = np.array([P[m].sum() for m in masks], dtype=np.float64)
            if stat == 'mean':
                band_vals = np.divide(band_vals, counts, out=np.zeros_like(band_vals), where=counts > 0)

            if per_band_norm:
                denom = band_vals.sum() + eps
                band_vals = band_vals / denom

            row.extend(band_vals)
        feats[i] = row

    return feats

