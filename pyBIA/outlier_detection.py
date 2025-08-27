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

class Classifier:
    """
    Build and apply an outlier-detection classifier on image cutouts.

    The classifier workflow supports optional min–max normalization, feature
    extraction (HOG, LBP, FFT, Wavelet, or simple statistics), optional
    imputation of missing values, and model fitting using an Isolation Forest
    (`clf='iforest'`). Multi-channel inputs are supported (features are computed
    per channel and concatenated), up to three channels.

    Parameters
    ----------
    data : ndarray or None, optional
        Image tensor with shape (N, H, W, C), where N is the number of samples,
        H×W are spatial dimensions, and C is the number of channels (C ≤ 3).
        Default is None (set later).
    normalize : bool, optional
        If True, min–max normalize each image/channel before feature extraction.
        Default is False.
    min_pixel : float, optional
        Lower bound for min–max normalization (used only if `normalize=True`).
        Default is 0.
    max_pixel : float, optional
        Upper bound for min–max normalization (used only if `normalize=True`).
        If multi-channel, the value is broadcast per channel. Default is 10.
    img_num_channels : int, optional
        Number of channels in the input tensor (last dimension). Must be set
        explicitly for legacy compatibility. Default is 1.
    feat_set : {'hog','lbp','fft','wavelet','stats'}, optional
        Feature family to compute for training. Default is 'hog'.
    clf : {'iforest'}, optional
        Classifier to train. Currently only Isolation Forest is supported.
        Default is 'iforest'.
    impute : bool, optional
        If True, impute missing feature values before fitting/predicting.
        Default is True.
    imp_method : {'knn','mean','median','mode','constant'}, optional
        Imputation strategy used by `impute_missing_values`. Default is 'knn'.
    SEED_NO : int, optional
        Random seed used for model initialization. Default is 1909.

    Attributes
    ----------
    model : IsolationForest or None
        Trained model after `create()`.
    imputer : object or None
        Fitted imputer returned by `impute_missing_values` when `impute=True`.
    data_x : ndarray
        Feature matrix derived from `data` after preprocessing and extraction.
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
        SEED_NO=1909
        ):

        self.data = data
        self.normalize = normalize
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.img_num_channels = img_num_channels
        self.feat_set = feat_set
        self.clf = clf
        self.impute = impute
        self.imp_method = imp_method
        self.SEED_NO = SEED_NO

        self.model = None
        self.imputer = None

        if feat_set not in ('hog', 'lbp', 'fft', 'wavelet', 'stats'):
            raise ValueError('The `feat_set` input is invalid, options are: "hog", "lbp", "fft", "wavelet", "stats"')

    def create(self):
        """
        Initialize, featurize, (optionally) impute, and fit the classifier.

        This method:
        1) Instantiates the requested model (Isolation Forest).
        2) Optionally normalizes `self.data` using min–max bounds.
        3) Extracts features according to `self.feat_set`.
        4) Replaces ±inf with NaN, then optionally imputes missing values.
        5) Fits the model on the resulting feature matrix.

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
        
        if self.clf == 'iforest':
            self.model = IsolationForest(random_state=self.SEED_NO)
        else:
            raise ValueError('Only IsolationForest is currently supported! Set `clf`="iforest" and run again.')
        
        if self.normalize:
            self.data = process_class(
                self.data, 
                normalize=self.normalize, 
                min_pixel=self.min_pixel, 
                max_pixel=[self.max_pixel]*self.img_num_channels, 
                img_num_channels=self.img_num_channels
                )
        
        if self.feat_set == 'hog':
            self.data_x = hog_feature_extraction(self.data)
        elif self.feat_set == 'lbp':
            self.data_x = lbp_feature_extraction(self.data)
        elif self.feat_set == 'wavelet':
            self.data_x = wavelet_energy_feature_extraction(self.data)
        elif self.feat_set == 'fft':
            self.data_x = fft_energy_feature_extraction(self.data)
        elif self.feat_set == 'stats':
            self.data_x = statistical_feature_extraction(self.data)

        self.data_x[np.isinf(self.data_x)] = np.nan

        if self.impute is False:

            if np.any(np.isfinite(self.data_x) == False):
                raise ValueError('data_x array contains nan values but `impute` is set to False! Set `impute`=True and run again.')
                        
            self.model.fit(self.data_x)            
            print(f"Returning base {self.clf} model...")

            return

        self.data_x, self.imputer = impute_missing_values(self.data_x, strategy=self.imp_method)
        
        self.model.fit(self.data_x)
                        
        print(f"Returning base {self.clf} model...")

        return
        
    def save(self, dirname=None, path=None, overwrite=False):
        """
        Save the trained model (and imputer) to disk.

        Creates a directory `pyBIA_outlier_model` under `path[/dirname]/` and
        writes the IsolationForest model and the fitted imputer (if applicable).

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

        if self.model is None and self.imputer is None and self.feats_to_use is None:
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
                try:
                    os.rmdir(path+'pyBIA_outlier_model')
                except OSError:
                    for file in os.listdir(path+'pyBIA_outlier_model'):
                        os.remove(path+'pyBIA_outlier_model/'+file)
                    os.rmdir(path+'pyBIA_outlier_model')
                os.mkdir(path+'pyBIA_outlier_model')
            else:
                raise ValueError('Tried to create "pyBIA_outlier_model" directory in specified path but folder already exists! If you wish to overwrite set `overwrite`=True.')
        
        path += 'pyBIA_outlier_model/'
        if self.model is not None:
            joblib.dump(self.model, path+'Model')
        if self.imputer is not None:
            joblib.dump(self.imputer, path+'Imputer')

        print(f'Files saved in: {path}')

        self.path = path

        return 

    def load(self, path=None):
        """ 
        Load a saved model/imputer from disk.

        Looks for a folder named `pyBIA_outlier_model` under `path` (or the user’s
        home directory if `path` is None) and attempts to load `Model` and `Imputer`
        artifacts into `self.model` and `self.imputer`.

        Parameters
        ----------
        path : str or None, optional
            Base directory containing `pyBIA_outlier_model/`. If None, uses the
            user's home directory.

        Returns
        -------
        None
        """

        path = str(Path.home()) if path is None else path 
        path = path+'/' if path[-1] != '/' else path 
        path += 'pyBIA_outlier_model/'

        try:
            self.model = joblib.load(path+'Model')
            model = 'model'
        except FileNotFoundError:
            model = ''
            pass

        try:
            self.imputer = joblib.load(path+'Imputer')
            imputer = ', imputer'
        except FileNotFoundError:
            imputer = ''
            pass 

        print('Successfully loaded the following class attributes: {}{}'.format(model, imputer))
        
        self.path = path

        return

    def predict(self, data):
        """
        Predict outlier/inlier labels and anomaly scores for new data.

        The input images are optionally normalized, featurized using the same
        `feat_set` as training, imputed (if an imputer was fitted), and passed
        to the trained Isolation Forest. Returns labels and scores.

        Parameters
        ----------
        data : ndarray
            Image tensor with shape (N, H, W, C). If multi-channel, features are
            computed per channel and concatenated (≤ 3 channels supported).

        Returns
        -------
        out : ndarray, shape (N, 3)
            Columns are:
              - `predictions`          : int, 1 for inlier, -1 for outlier
              - `decision_function`    : float, signed anomaly score (higher is more inlier)
              - `raw_anomaly_scores`   : float, `decision_function + self.model.offset_`

        Raises
        ------
        ValueError
            If `create()` has not been called (no trained model).
        ValueError
            If the feature matrix contains NaN/inf and no `imputer` is available.
        """

        if self.model is None:
            raise ValueError('No `model` has been created! Run the create() method first!')

        if self.normalize:
            data = process_class(
                data, 
                normalize=self.normalize, 
                min_pixel=self.min_pixel, 
                max_pixel=[self.max_pixel]*self.img_num_channels, 
                img_num_channels=self.img_num_channels
                )
        
        if self.feat_set == 'hog':
            data_x = hog_feature_extraction(data)
        elif self.feat_set == 'lbp':
            data_x = lbp_feature_extraction(data)
        elif self.feat_set == 'wavelet':
            data_x = wavelet_energy_feature_extraction(data)
        elif self.feat_set == 'fft':
            data_x = fft_energy_feature_extraction(data)
        elif self.feat_set == 'stats':
            data_x = statistical_feature_extraction(data)

        if self.imputer is None:
            if np.any(np.isfinite(data_x) == False):
                print(self.feat_set, data_x)
                raise ValueError('data_x array contains nan values but `impute` is set to False! Was `impute`=True when the model was created? If so, set `impute`=True and run again.')
        else:
            data_x = self.imputer.transform(data_x)

        if len(data_x.shape) == 1:
            data_x = data_x.reshape(1, -1)

        decision_function_scores = self.model.decision_function(data_x)
        raw_anomaly_scores = decision_function_scores + self.model.offset_
        predictions = np.where(decision_function_scores < 0, -1, 1) #If the score is < 0 set -1 (outlier) other 1 (inlier)
        
        return np.c_[predictions, decision_function_scores, raw_anomaly_scores]


def hog_feature_extraction(
    images, 
    return_image=False, 
    max_pool=False
    ):
    """
    Extract Histogram of Oriented Gradients (HOG) features per channel.

    For each image and channel, HOG features are computed using scikit-image
    defaults (grayscale per channel). Channel-wise features are then either
    concatenated (default) or reduced to a single scalar via max pooling.
    Optionally, HOG visualizations are returned (one per channel).

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
    - HOG parameters are the scikit-image defaults (orientations, pixels per
      cell, cells per block, block normalization).
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

    For each image channel, a 2D decimated DWT (`pywt.wavedec2`) is computed up to
    level `L`. The feature vector per channel is the energy of the approximation
    band at level `L` followed by the energies of the detail bands (H, V, D) for
    levels `L..1`: `[A_L, (H,V,D)_L, …, (H,V,D)_1]`. With an orthogonal wavelet
    (e.g., 'db4') and symmetric extension, these energies correspond to L2 power
    per subband. Per-image features are formed by concatenating all channel vectors.

    Parameters
    ----------
    images : sequence of ndarray or ndarray
        Iterable of images with shape `(H, W, C)` **or** an array with shape
        `(N, H, W, C)`. Iteration is over the first dimension.
    wavelet : str, optional
        Wavelet name for PyWavelets (e.g., 'db4'). Default is 'db4'.
    level : int or None, optional
        Decomposition level `L`. If None, uses the maximum level allowed by the
        image size and wavelet filter length. Default is None.
    mode : str, optional
        Boundary extension mode passed to `pywt.wavedec2`. Default is 'symmetric'.
    stat : {'sum','mean'}, optional
        Aggregation for each subband:
        - 'sum'  : sum of squares (energy)
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
            sizes    = []

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
            sizes    = np.asarray(sizes, dtype=float)

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
        9) skewness  (scipy.stats.skew, bias=True)
       10) kurtosis  (scipy.stats.kurtosis, fisher=True, bias=True)

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

    For each image channel, canonical LBP codes are computed with
    `skimage.feature.local_binary_pattern(method='default')` using `P`
    sampling points on a circle of radius `R`. A histogram with `2**P`
    bins (codes 0..2**P−1) is computed and L1-normalized (`density=True`).
    Per-image feature vectors are formed by concatenating the channel
    histograms.

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

    For each image/channel, the 2D power spectrum is computed and integrated over
    Nyquist-normalized radial annuli defined by `band_edges` (0 → DC, 1 → per-axis
    Nyquist = 0.5 cyc/pixel). Optionally apply a separable Hann window to reduce
    spectral leakage, remove the DC component, and normalize band energies to sum
    to one per channel.

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

