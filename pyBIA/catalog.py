# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:10:11 2021

@author: danielgodinez
"""
from pathlib import Path
from contextlib import suppress
from warnings import filterwarnings, warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
from matplotlib.patches import Patch
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.utils.exceptions import AstropyWarning
from astropy.convolution import Gaussian2DKernel, convolve
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources, SourceCatalog
from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus
from progress import bar 

from pyBIA import data_processing
from pyBIA.image_moments import make_moments_table

with suppress(ModuleNotFoundError):
    import scienceplots
    plt.style.use("science")
    plt.rcParams.update({"font.size": 21})

filterwarnings("ignore", category=AstropyWarning)
filterwarnings("ignore", category=RuntimeWarning)


class Catalog:
    """
    Build photometric and morphological catalogs from postage-stamp astronomical images.

    This class extracts source positions and performs aperture photometry along with
    segmentation-based morphological analysis. Sources can be detected automatically
    via image segmentation or specified manually through input coordinates. The resulting
    catalog includes flux measurements, optional background subtraction, and a comprehensive
    set of shape descriptors for use in classification pipelines.

    Parameters
    ----------
    data : ndarray
        2D image array.
    x, y : array-like or None, optional
        Pixel coordinates of source centers. If None, sources are detected automatically.
    bkg : float or None, optional
        Background mode. Use 0 if background is already subtracted; None to estimate
        the local sky background.
    error : ndarray or None, optional
        Pixel-wise error map with the same shape as `data`.
    zp : float or None, optional
        Zeropoint for magnitude calculations. If None, magnitudes are not computed.
    exptime : float or None, optional
        Exposure time in seconds. If provided, `data` is normalized (e.g., counts per sec)
        when performing segmentation and computing morphology.
    morph_params : bool, optional
        If True, compute moment-based morphological features (default is True).
    nsig : float, optional
        Detection threshold in units of background sigma. Pixels above `nsig` are
        considered in segmentation. Default is 0.3.
    threshold : int, optional
        Radius (in pixels) around the source center used to validate detection.
        If no object is found within this region, the source is flagged as a non-detection.
        Set to 0 to require exact overlap. Default is 10.
    deblend : bool, optional
        If True, enables deblending of overlapping sources. Default is False
        (recommended for diffuse, extended objects).
    obj_name : array-like or None, optional
        List of object names for catalog rows.
    field_name : array-like or None, optional
        List of field names for catalog rows.
    flag : array-like or None, optional
        List of flags for catalog rows.
    aperture : int, optional
        Aperture radius (in pixels) for photometry. Default is 15.
    annulus_in : int, optional
        Inner radius (in pixels) of background annulus for local sky estimation.
        Default is 20.
    annulus_out : int, optional
        Outer radius (in pixels) of background annulus. Default is 35.
    kernel_size : int, optional
        Size of Gaussian kernel (in pixels) for segmentation smoothing. Default is 21.
    npixels : int, optional
        Minimum area (in pixels) for segmentation detection. Default is 9.
    connectivity : int, optional
        Pixel connectivity for segmentation (4 or 8). Default is 8.
    invert : bool, optional
        If True, flips the (x, y) input order when cropping sub-images.
        Useful for data with (row, column) indexing or FITS-style origin.
        Default is False.
    cat : pandas.DataFrame or None, optional
        Existing catalog to augment or use for metadata.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        x: np.ndarray | list | None = None,
        y: np.ndarray | list | None = None,
        bkg: float | None = None,
        error: np.ndarray | None = None,
        zp: float | None = None,
        exptime: float | None = None,
        morph_params: bool = True,
        nsig: float = 0.3,
        threshold: int = 10,
        deblend: bool = False,
        obj_name=None,
        field_name=None,
        flag=None,
        aperture: int = 15,
        annulus_in: int = 20,
        annulus_out: int = 35,
        kernel_size: int = 21,
        npixels: int = 9,
        connectivity: int = 8,
        invert: bool = False,
        cat: pd.DataFrame | None = None,
    ):
        # Data
        self.data = data
        self.error = error
        self.zp = zp
        self.exptime = exptime

        # Source detection and photometry
        self.morph_params = morph_params
        self.nsig = nsig
        self.threshold = threshold
        self.deblend = deblend
        self.aperture = aperture
        self.annulus_in = annulus_in
        self.annulus_out = annulus_out
        self.kernel_size = kernel_size
        self.npixels = npixels
        self.connectivity = connectivity
        self.invert = invert
        self.bkg = bkg

        # The catalog, can be input to extract obj_name, field_name and object flag (may remove in future versions)
        self.cat = cat

        # Source positions and field info
        self.x = None if x is None else np.atleast_1d(x)
        self.y = None if y is None else np.atleast_1d(y)
        self.obj_name = None if obj_name is None else np.atleast_1d(obj_name)
        self.field_name = None if field_name is None else np.atleast_1d(field_name)
        self.flag = None if flag is None else np.atleast_1d(flag)

        # if existing catalog given, use available data 
        if cat is not None:
            for key in ("obj_name", "field_name", "flag"):
                with suppress(KeyError):
                    setattr(self, key, np.array(cat[key]))

    def create(
        self, 
        *, 
        save_file: bool = True, 
        path: str | None = None, 
        filename: str | None = None
        ):
        """
        Build the full photometric and morphological catalog.

        This method performs source detection (via segmentation or user-supplied positions),
        computes aperture photometry, and optionally includes morphological features. It
        returns the catalog as a pandas DataFrame and can also save it to disk.

        Parameters
        ----------
        save_file : bool, optional
            If True, saves the catalog as a CSV file. Default is True.
        path : str or None, optional
            Directory where the output CSV file will be saved. If None, defaults to the home directory.
        filename : str or None, optional
            Name of the output CSV file. Defaults to "pyBIA_catalog.csv".

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the photometric and morphological measurements for all detected sources.

        Raises
        ------
        ValueError
            If background mode (`bkg`) is invalid, `data` and `error` shapes mismatch,
            aperture and annulus sizes are incompatible, or if `x` and `y` coordinates differ in length.

        Notes
        -----
        - If `x` and `y` are not provided, automatic source detection is run on the full frame.
        - If `x` and `y` are given, photometry and morphology are computed only at those positions.
        - Catalog output includes flux, flux error, optional magnitudes, and morphology features
          depending on initialization settings.
        """

        # Input checks
        if self.bkg not in (None, 0):
            raise ValueError("If data are background-subtracted set bkg=0; otherwise use bkg=None to estimate local sky.")
        if self.error is not None and self.data.shape != self.error.shape:
            raise ValueError("`error` must match shape of `data`.")
        if self.aperture >= self.annulus_in or self.annulus_in >= self.annulus_out:
            raise ValueError("Must satisfy aperture < annulus_in < annulus_out.")
        if (self.x is not None) and (len(self.x) != len(self.y)):
            raise ValueError("`x` and `y` must be same length.")
        if not (isinstance(self.threshold, (int, float))):
            raise TypeError('The `threshold` parameter must be >= 0! Set to 0 if a detection must be present at the specific position(s).')

        # Source detection 
        if self.x is None:
            self._auto_detect_sources()
        else:
            self._aperture_photometry()

        # Save cat
        if save_file:
            path = Path(path) if path is not None else Path.home()
            filename = filename or "pyBIA_catalog.csv"
            self.cat.to_csv(path / filename, index=False)

        return self.cat
 
    def _auto_detect_sources(self):
        """
        Automatically detect sources using segmentation and build the catalog.

        This method performs full-frame source detection via image segmentation
        (using `photutils.detect_sources`), estimates background if needed, computes
        aperture photometry, and optionally derives morphological features using 
        moment-based shape descriptors. Results are stored in `self.cat`.
        """

        if self.nsig > 1 and not self.deblend:
            warn("Very high `nsig`; consider lowering or enabling `deblend`.")

        # Subtract background if data is not yet background-subtracted (e.g., bkg=None)
        self.data_bgsub = self._subtract_global_background() if self.bkg is None else self.data

        # Detect sources using the image segmentation routine from Astropy (photutils.detect_sources)
        segm, conv = segm_find(
            self.data_bgsub, nsig=self.nsig, kernel_size=self.kernel_size,
            deblend=self.deblend, npixels=self.npixels,
            connectivity=self.connectivity,
        )

        # Generate the source catalog 
        props = SourceCatalog(self.data_bgsub, segm, convolved_data=conv)
        #centroids = np.asarray(props.centroid)
        #self.x, self.y = centroids[:, 0], centroids[:, 1]
        try:
            self.x, self.y = props.centroid[:,0], props.centroid[:,1]
        except:
            self.x, self.y = props.centroid[0], props.centroid[1]
        print(f"{len(self.x)} sources detected.")

        # photometry
        positions = list(zip(self.x, self.y))
        aper_stats = ApertureStats(self.data_bgsub, CircularAperture(positions, r=self.aperture), error=self.error)
        flux_err = None if self.error is None else aper_stats.sum_err

        # morphological params
        if self.morph_params:
            props_list, moments, self.segm_map = morph_parameters(
                self.data_bgsub, self.x, self.y, exptime=self.exptime,
                nsig=self.nsig, kernel_size=self.kernel_size,
                npixels=self.npixels, connectivity=self.connectivity, 
                median_bkg=None, invert=self.invert, deblend=self.deblend, 
                threshold=self.threshold
            )
            tbl = make_table(props_list, moments)
        else:
            tbl = None

        self.cat = make_dataframe(
            table=tbl, x=self.x, y=self.y, zp=self.zp,
            obj_name=self.obj_name, field_name=self.field_name, flag=self.flag,
            flux=aper_stats.sum, flux_err=flux_err, median_bkg=None
        )
 
    def _aperture_photometry(self):
        """
        Perform aperture photometry at user-supplied positions and build the catalog.

        This method computes circular-aperture fluxes at specified `(x, y)` coordinates,
        optionally subtracts a local background estimated from an annular region, and
        calculates flux errors if an error map is provided. Morphological features are
        computed if enabled. The resulting catalog is stored in `self.cat`.      
        """

        positions = list(zip(self.x, self.y))
        apertures = CircularAperture(positions, r=self.aperture)
        aper_stats = ApertureStats(self.data, apertures, error=self.error)

        # local background per source
        if self.bkg is None:
            ann = CircularAnnulus(positions, r_in=self.annulus_in, r_out=self.annulus_out)
            bkg_stats = ApertureStats(self.data, ann, error=self.error, sigma_clip=SigmaClip())
            bkg = bkg_stats.median
            flux = aper_stats.sum - bkg * apertures.area
        else: # if data is already backgroud-subtracted 
            bkg, flux = None, aper_stats.sum

        # morph params
        if self.morph_params:
            props_list, moments, self.segm_map = morph_parameters(
                self.data, self.x, self.y, exptime=self.exptime,
                nsig=self.nsig, kernel_size=self.kernel_size,
                npixels=self.npixels, connectivity=self.connectivity,
                median_bkg=bkg, invert=self.invert, deblend=self.deblend,
                threshold=self.threshold,
            )
            tbl = make_table(props_list, moments)
        else:
            tbl = None

        # Set explicitly because aper_stats.sum_err will return nan if no error map is input
        flux_err = None if self.error is None else aper_stats.sum_err

        self.cat = make_dataframe(
            table=tbl, x=self.x, y=self.y, zp=self.zp,
            obj_name=self.obj_name, field_name=self.field_name, flag=self.flag,
            flux=flux, flux_err=flux_err, median_bkg=bkg,
        )

    def _subtract_global_background(self):
        """
        Estimate and subtract a global background level from the image. Only used when no catalog positions are input.

        For large images, the method estimates the background using a sliding box 
        of size `2 × annulus_out`, ensuring the region encompasses the largest 
        background annulus used for photometry. For small images, a single 
        sigma-clipped median value is subtracted instead.

        Returns
        -------
        ndarray
            Background-subtracted image.
        """

        length = self.annulus_out * 2 * 2 #The sub-array when padding will be a square encapsulating the outer annuli
        
        if (self.data.shape[0] < length) or (self.data.shape[1] < length):
            bg = sigma_clipped_stats(self.data)[1]
            return self.data - bg
        
        return subtract_background(self.data, length=length)


def morph_parameters(
    data, 
    x, 
    y, 
    size=100, 
    nsig=0.6, 
    threshold=10, 
    kernel_size=21, 
    median_bkg=None, 
    invert=False, 
    deblend=False, 
    exptime=None, 
    npixels=9, 
    connectivity=8):
    """
    Compute segmentation-based morphological features for sources at given positions.

    For each (x, y) position, a `size × size` cutout is extracted, optionally background-
    corrected and exposure-normalized, then segmented (via `segm_find`) to isolate the
    central source. Moment-based features are measured (via `make_moments_table`) and
    photutils-like source properties are recorded. Results are suitable for downstream
    classification tasks.

    Parameters
    ----------
    data : ndarray
        2D image array.
    x, y : array-like or scalar
        Pixel coordinates of source centers. Scalars are accepted and will be promoted
        to length-1 arrays.
    size : int, optional
        Side length (pixels) of the square cutout used per source. If the image is
        smaller than `size` along any axis, the largest square that fits is used.
        Default is 100.
    nsig : float, optional
        Detection threshold in units of background sigma for segmentation. Pixels
        above `nsig` contribute to detected regions. Default is 0.6.
    threshold : int, optional
        Central-detection validation radius (pixels). If `threshold == 0`, require
        that the exact central pixel belongs to a segmented object; otherwise, require
        at least one segmented pixel within a central circular region of radius
        `threshold`. Sources failing this test are flagged as non-detections.
        Default is 10.
    kernel_size : int, optional
        Gaussian kernel size (pixels) used by `segm_find` for segmentation smoothing.
        Default is 21.
    median_bkg : array-like or None, optional
        Per-source median background values to subtract from each cutout (not a full
        background map). If None, input `data` is assumed to be background-subtracted.
        Length must match `x`/`y` if provided. Default is None.
    invert : bool, optional
        If True, swap (x, y) when cropping (useful for data in row–column indexing
        or FITS-style top-left origin). Default is False.
    deblend : bool, optional
        If True, enable deblending in `segm_find` to split overlapping sources.
        Default is False.
    exptime : float or None, optional
        Exposure time in seconds. If provided, each cutout is divided by `exptime`
        prior to segmentation/feature measurement (e.g., to convert counts to counts/s).
        Default is None.
    npixels : int, optional
        Minimum area (pixels) for valid segmented regions. Default is 9.
    connectivity : int, optional
        Pixel connectivity for segmentation (4 or 8). Default is 8.

    Returns
    -------
    props_list : ndarray of object
        Array of per-source photutils-like property selections (e.g., slices from
        `SourceCatalog`). For non-detections, the sentinel value `-999` is inserted.
    moments_list : list of pandas.DataFrame
        Per-source moment feature tables returned by `make_moments_table`. For
        non-detections, the sentinel value `-999` is inserted.
    segm_map : ndarray
        Segmentation map (int labels) for the last processed cutout. If any source
        failed detection, a zero array of that cutout's shape is returned.

    Raises
    ------
    ValueError
        If the number of properties and moment tables differs (internal consistency check).

    Notes
    -----
    - Each source is processed independently using a cropped cutout for computational efficiency.
    - Central-detection validation:
        * `threshold == 0` enforces exact central-pixel membership in a segment.
        * `threshold > 0` accepts any segment intersecting the central circular mask.
      The segment nearest the center is retained when multiple intersect the mask.
    - FITS coordinate convention: Many FITS images use a top-left origin (row, column).
      If your coordinates follow this convention, set `invert=True` so cropping treats
      inputs correctly. `pyBIA` assumes standard (x to the right, y upward) unless inverted.
    - For very small images (`< ~50` pixels on a side), results may be unstable if the
      source is truncated at the edges (a warning is printed).
    """

    if data.shape[0] < 50:
        print('Small image warning: results may be unstable if the object does not fit entirely within the frame.')
    if not isinstance(x, (list, tuple, np.ndarray)):
        x, y = [x], [y]

    size = size if data.shape[0] > size and data.shape[1] > size else min(data.shape[0],data.shape[1])

    prop_list, moment_list = [], []
    progress_bar = bar.FillingSquaresBar('Applying image segmentation...', max=len(x))

    for i in range(len(x)):
        new_data = data_processing.crop_image(data, int(x[i]), int(y[i]), size, invert=invert)
        if median_bkg is not None:
            new_data -= median_bkg[i] 
        if exptime is not None:
            new_data /= exptime
       
        segm, convolved_data = segm_find(new_data, nsig=nsig, kernel_size=kernel_size, deblend=deblend, npixels=npixels, connectivity=connectivity)
        try:
            props = SourceCatalog(new_data, segm, convolved_data=convolved_data)
        except: #If there are no segmented objects in the image
            prop_list.append(-999)
            moment_list.append(-999)
            progress_bar.next()
            continue

        # If user requires central source detection but nothing is present!
        if threshold == 0:
            if segm.data[size//2, size//2] == 0: # If no segmentation patch is in the center it is considered a non-detection
                prop_list.append(-999)
                moment_list.append(-999)
                progress_bar.next()
                continue
            else:
                segm_label = segm.data[size//2, size//2] # The patch that goes over the center
                new_data[segm.data != segm_label] = 0
                inx = np.where(props.label == segm_label)[0]
                prop_list.append(props[inx])
        else:
            # Mask a circular area at the center of the image, using radius=threshold
            # Flag if there is no segmented object within the circular mask 
            rr, cc = np.ogrid[:size, :size]
            cx = cy = size // 2
            mask = (rr - cx) ** 2 + (cc - cy) ** 2 <= threshold**2
            """
            labels_in_mask = np.unique(segm.data[mask])
            labels_in_mask = labels_in_mask[labels_in_mask != 0]  # drop background
            if labels_in_mask.size == 0:
                prop_list.append(-999)
                moment_list.append(-999)
                progress_bar.next()
                continue

            # map those labels to indices in props
            idxs = [np.where(props.label == lab)[0][0] for lab in labels_in_mask]
            # pick the intersecting object whose centroid is closest to center
            dists = [np.hypot(float(props[i].centroid[0]) - cx, float(props[i].centroid[1]) - cy) for i in idxs]
            inx = np.array([idxs[int(np.argmin(dists))]])
            new_data[segm.data != props[inx].label] = 0
            prop_list.append(props[inx])
            """
            if np.count_nonzero(segm.data[mask]) == 0: 
                prop_list.append(-999)
                moment_list.append(-999)
                progress_bar.next()
                continue

            #This is to select the segmented object closest to the center, (x,y)=(size/2, size/2)
            separations = [np.hypot(float(p.centroid[0]) - cx, float(p.centroid[1]) - cy) for p in props]
            inx = np.array([np.argmin(separations)])

            new_data[segm.data != props[inx].label] = 0
            prop_list.append(props[inx])

        ##### Image Moments (Our own computations) #####
        moments_table = make_moments_table(new_data)
        moment_list.append(moments_table)
        progress_bar.next()
    progress_bar.finish()

    if len(prop_list) != len(moment_list):
        raise ValueError('The properties list does not match the image moments list.')
    
    if -999 in prop_list:
        print('NOTE: At least one object could not be detected in segmentation, perhaps the object is too faint. The morphological features have been set to -999.')
        return np.array(prop_list, dtype=object), moment_list, np.zeros(new_data.shape)
    else:
        return np.array(prop_list, dtype=object), moment_list, segm.data

def make_table(props, moments):
    """
    Assemble a flat feature array from photutils `SourceCatalog` properties and custom moments.

    For each source, this function concatenates (i) the moment-based features from
    `moments` and (ii) a selected subset of photutils segmentation properties from
    `props`. The resulting per-source feature vector is suitable for ML pipelines.

    Parameters
    ----------
    props : sequence
        Sequence (length N) where each element is an indexable selection of a
        photutils `SourceCatalog` (e.g., `props[i][0]`) representing the segmented
        source for sample i. If a source is missing (e.g., no segment), that entry
        should still exist but may be unusable; this function will emit sentinels.
    moments : sequence
        Sequence (length N) of mapping-like objects (e.g., dict or DataFrame row)
        containing scalar moment features keyed by the following names:
        `["M00","M10","M01","M20","M11","M02","M30","M21","M12","M03",
          "mu00","mu10","mu01","mu20","mu11","mu02","mu30","mu21","mu12","mu03",
          "G00","G10","G01","G20","G11","G02","G30","G21","G12","G03",
          "Hu1","Hu2","Hu3","Hu4","Hu5","Hu6","Hu7",
          "L00","L10","L01","L20","L11","L02","L30","L21","L12","L03"]`.

    Returns
    -------
    features : ndarray, shape (N, D)
        Per-source feature matrix. 

    Notes
    -----
    - If a source has no valid segmented object or moments, the function fills the
      entire feature vector for that source with the sentinel value `-999`.
    - The `'isscalar'` property is cast to an integer flag: 1 for True (single source),
      0 for False.
    - All other properties are extracted as scalars from the photutils table.
    """

    moment_list = ["M00","M10","M01","M20","M11","M02","M30","M21","M12","M03",
        "mu00","mu10","mu01","mu20","mu11","mu02","mu30","mu21","mu12","mu03",
        "G00","G10","G01","G20","G11","G02","G30","G21","G12","G03",
        "Hu1","Hu2","Hu3","Hu4","Hu5","Hu6","Hu7",
        "L00","L10","L01","L20","L11","L02","L30","L21","L12","L03"]

    prop_list = ['area', 'covar_sigx2', 'covar_sigy2', 'covar_sigxy', 'covariance_eigvals', 
        'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 
        'fwhm', 'gini', 'orientation', 'perimeter', 'semimajor_sigma', 'semiminor_sigma', 
        'isscalar', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'max_value', 'maxval_xindex', 
        'maxval_yindex', 'min_value', 'minval_xindex', 'minval_yindex']
    
    table = []
    print('Writing catalog...')
    for i in range(len(props)):

        morph_feats = []

        try:
            props[i][0].area #To avoid when this is None
            for moment in moment_list:
                morph_feats.append(float(moments[i][moment]))
        except:
            for j in range(len(prop_list+moment_list) + 1): #+1 because the covariance_eigvals represents the 2 eigenvalues of the covariance matrix
                morph_feats.append(-999)
            table.append(morph_feats)
            continue

        QTable = props[i][0].to_table(columns=prop_list)
        for param in prop_list:
            if param == 'covariance_eigvals': 
                morph_feats.append(np.ravel(QTable[param])[1].value) #This is the first eigval
                morph_feats.append(np.ravel(QTable[param])[0].value) #This is the second eigval
            elif param == 'isscalar':
                if QTable[param]: #Checks whether it's a single source, 1 for true, 0 for false
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)
            else:
                morph_feats.append(QTable[param].value[0])

        table.append(morph_feats)

    return np.array(table, dtype=object)


def make_dataframe(
    table=None, 
    x=None, 
    y=None, 
    zp=None, 
    flux=None,
    flux_err=None, 
    median_bkg=None, 
    obj_name=None,
    field_name=None, 
    flag=None, 
    save=True,
    path=None, 
    filename=None
    ):
    """
    Assemble a photometry+morphology catalog into a pandas DataFrame (and optional CSV).

    This function merges per-source metadata (names, positions, flags), photometric
    measurements (flux, flux error, optional magnitudes), background statistics, and
    morphology features (moments + photutils properties) into a single DataFrame. If
    requested, the table is also written to disk as a CSV.

    Parameters
    ----------
    table : array-like or None, optional
        Feature matrix from `make_table`, shape (N, D) or (D,) for a single source.
        Columns are expected to be ordered as:
        moments ["M00","M10","M01","M20","M11","M02","M30","M21","M12","M03",
                 "mu00","mu10","mu01","mu20","mu11","mu02","mu30","mu21","mu12","mu03",
                 "G00","G10","G01","G20","G11","G02","G30","G21","G12","G03",
                 "Hu1","Hu2","Hu3","Hu4","Hu5","Hu6","Hu7",
                 "L00","L10","L01","L20","L11","L02","L30","L21","L12","L03"]
        followed by photutils properties
        ["area","covar_sigx2","covar_sigy2","covar_sigxy",
         "covariance_eigval1","covariance_eigval2",
         "cxx","cxy","cyy","eccentricity","ellipticity","elongation",
         "equivalent_radius","fwhm","gini","orientation","perimeter",
         "semimajor_sigma","semiminor_sigma","isscalar",
         "bbox_xmax","bbox_xmin","bbox_ymax","bbox_ymin",
         "max_value","maxval_xindex","maxval_yindex",
         "min_value","minval_xindex","minval_yindex"].
        Default is None (no morphology columns added).
    x, y : array-like or None, optional
        Pixel coordinates of source centers. If provided, columns `xpix`, `ypix`
        are added. Length should match the number of rows N. Default is None.
    zp : float or None, optional
        Photometric zeropoint. If provided with `flux`, columns `mag` and `mag_err`
        are computed as `-2.5*log10(flux) + zp` and `(2.5/ln 10)*(flux_err/flux)`,
        respectively. Default is None.
    flux : array-like or None, optional
        Aperture-sum fluxes; adds column `flux`. Default is None.
    flux_err : array-like or None, optional
        Flux uncertainties; adds column `flux_err`. If `zp` is also provided,
        `mag_err` is computed. Default is None.
    median_bkg : array-like or None, optional
        Per-source median background values; adds column `median_bkg`. Default is None.
    obj_name : array-like or None, optional
        Per-source object names; adds column `obj_name`. Default is None.
    field_name : array-like or None, optional
        Per-source field names; adds column `field_name`. Default is None.
    flag : array-like or None, optional
        Per-source flag values; adds column `flag`. Default is None.
    save : bool, optional
        If True, write the DataFrame to CSV. Default is True.
    path : str or Path or None, optional
        Output directory for CSV. If None, use the user's home directory. Default is None.
    filename : str or None, optional
        Output CSV filename. Default is "pyBIA_catalog.csv".

    Returns
    -------
    df : pandas.DataFrame
        Catalog with available columns among:
        - Metadata: `obj_name`, `field_name`, `flag`
        - Positions: `xpix`, `ypix`
        - Background: `median_bkg`
        - Photometry: `flux`, `flux_err`, and (if `zp` provided) `mag`, `mag_err`
        - Morphology: the full set listed in `table` (moments + photutils properties)

    Notes
    -----
    - If `table` is 1D, it is promoted to 2D with a single row.
    - Magnitudes are computed only when both `flux` and `zp` are provided; no
      guards are applied here for non-positive `flux` (users should prefilter or
      post-handle infinities/NaNs if needed).
    - When `save=True`, the CSV is written to `path/filename` with `index=False`.
    """

    filename = filename or "pyBIA_catalog.csv"

    # This combines the two lists in the make_table function but instead of covariance_eigvals 
    base_cols = ["M00","M10","M01","M20","M11","M02","M30","M21","M12","M03",
        "mu00","mu10","mu01","mu20","mu11","mu02","mu30","mu21","mu12","mu03",
        "G00","G10","G01","G20","G11","G02","G30","G21","G12","G03",
        "Hu1","Hu2","Hu3","Hu4","Hu5","Hu6","Hu7",
        "L00","L10","L01","L20","L11","L02","L30","L21","L12","L03",
        "area","covar_sigx2","covar_sigy2","covar_sigxy",
        "covariance_eigval1","covariance_eigval2",
        "cxx","cxy","cyy","eccentricity","ellipticity","elongation",
        "equivalent_radius","fwhm","gini","orientation","perimeter",
        "semimajor_sigma","semiminor_sigma","isscalar",
        "bbox_xmax","bbox_xmin","bbox_ymax","bbox_ymin",
        "max_value","maxval_xindex","maxval_yindex",
        "min_value","minval_xindex","minval_yindex"]

    # To store the catalog
    data_dict = {}

    if obj_name is not None: data_dict["obj_name"] = obj_name
    if field_name is not None:data_dict["field_name"] = field_name
    if flag is not None: data_dict["flag"] = flag
    if x is not None: data_dict["xpix"] = x
    if y is not None: data_dict["ypix"] = y
    if median_bkg is not None:data_dict["median_bkg"] = median_bkg
    if flux is not None:
        data_dict["flux"] = flux
        if zp is not None:
            data_dict["mag"] = -2.5 * np.log10(np.array(flux)) + zp
    if flux_err is not None:
        data_dict["flux_err"] = flux_err
        if zp is not None:
            data_dict["mag_err"] = (2.5/np.log(10)) * (np.array(flux_err)/np.array(flux))

    # build the df
    if table is not None:
        table = np.atleast_2d(table)
        for col, vals in zip(base_cols, table.T):
            data_dict[col] = vals

    df = pd.DataFrame(data_dict)

    if save:
        save_path = Path(path) if path else Path.home()
        df.to_csv(save_path / filename, index=False)

    return df
  
def subtract_background(data, length=150):
    """
    Subtract a local background estimate from a 2D image.

    The image is divided into non-overlapping square regions of size
    `length × length`. For each region, the sigma-clipped median pixel
    value is computed and subtracted from that region. For images whose
    dimensions are not divisible by `length`, the array is padded
    symmetrically so that tiles align evenly. Padding is removed before
    return.

    Parameters
    ----------
    data : ndarray
        2D image array.
    length : int, optional
        Side length (pixels) of local regions used for background estimation.
        Default is 150. Smaller values capture more local variations, while
        larger values enforce a smoother background.

    Returns
    -------
    data_sub : ndarray
        Background-subtracted image of the same shape as input.

    Notes
    -----
    - For small images (`min(data.shape) < length`), no tiling is done;
      instead, the global sigma-clipped median is subtracted.
    - Padding is applied symmetrically (`mode='symmetric'`) so that
      regions near the edges are treated consistently. Padding is sliced
      away before returning.
    - Background estimation uses `astropy.stats.sigma_clipped_stats`,
      which is robust against outliers.
    """

    Ny, Nx = data.shape

    if Nx < length or Ny < length: #Small image, no need to pad, just take robust median
        background  = sigma_clipped_stats(data)[1] #Sigma clipped median
        data -= background
        return data

    pad_x = length - (Nx % length) 
    pad_y = length - (Ny % length) 
    padded_matrix = np.pad(data, [(0, int(pad_y)), (0, int(pad_x))], mode='symmetric')
   
    x_increments = int(padded_matrix.shape[1] / length)
    y_increments = int(padded_matrix.shape[0] / length)

    initial_x, initial_y = int(length/2), int(length/2)
    x_range = [initial_x+length*n for n in range(x_increments)]
    y_range = [initial_y+length*n for n in range(y_increments)]

    positions=[]
    for xp in x_range:
        for yp in y_range:
            positions.append((xp, yp))

    for i in range(len(positions)):
        x,y = positions[i][0], positions[i][1]
        background  = sigma_clipped_stats(padded_matrix[int(y)-initial_y:int(y)+initial_y,int(x)-initial_x:int(x)+initial_x])[1] #Sigma clipped median                        
        padded_matrix[int(y)-initial_y:int(y)+initial_y,int(x)-initial_x:int(x)+initial_x] -= background

    data = padded_matrix[:-int(pad_y),:-int(pad_x)] #Slice away the padding 

    return data

def segm_find(
    data: np.ndarray, 
    *, 
    nsig: float = 0.6, 
    kernel_size: int = 21, 
    deblend: bool = False, 
    npixels: int = 9, 
    connectivity: int = 8
    ):
    """
    Perform image segmentation to detect sources above a sigma threshold.

    The input image is convolved with a 2D Gaussian kernel, then thresholded
    at `nsig × sigma` to identify sources. Optionally, overlapping sources can
    be deblended. Returns both the segmentation map and the convolved image.

    Parameters
    ----------
    data : ndarray
        2D background-subtracted image array.
    nsig : float, optional
        Detection threshold in units of background sigma. Pixels above
        `nsig` are considered during segmentation. Default is 0.6.
    kernel_size : int, optional
        Size (pixels) of the square Gaussian kernel used for convolution.
        Must be odd. Default is 21.
    deblend : bool, optional
        If True, deblend overlapping sources in the segmentation map.
        Default is False (recommended when preserving diffuse blobs as
        single objects).
    npixels : int, optional
        Minimum number of connected pixels above threshold required to
        define a source. Default is 9.
    connectivity : int, optional
        Pixel connectivity: 4 (edge-connected) or 8 (edge+corner-connected).
        Default is 8.

    Returns
    -------
    segm : `photutils.segmentation.SegmentationImage` or None
        Segmentation image labeling detected sources. None if no sources
        are found.
    convolved_data : ndarray
        Gaussian-convolved version of the input `data` used for source
        detection.

    Notes
    -----
    - Input `data` must be background-subtracted prior to calling this
      function.
    - The Gaussian kernel is constructed with FWHM = 9 pixels
      (`sigma = 9 × gaussian_fwhm_to_sigma`) and size `kernel_size × kernel_size`.
    - If `deblend=True`, `photutils.segmentation.deblend_sources` is applied
      to split overlapping sources.
    """

    threshold = detect_threshold(data, nsigma=nsig, background=0.0)
    sigma_pix = 9.0 * gaussian_fwhm_to_sigma   # FWHM = 9. smooth the data with a 2D circular Gaussian kernel with a FWHM of 3 pixels to filter the image prior to thresholding
    kernel = Gaussian2DKernel(sigma_pix, x_size=kernel_size, y_size=kernel_size, mode='center')
    convolved_data = convolve(data, kernel, normalize_kernel=True, preserve_nan=True)
    segm = detect_sources(convolved_data, threshold, npixels=npixels, connectivity=connectivity)
    if deblend and segm is not None:
        segm = deblend_sources(convolved_data, segm, npixels=npixels, connectivity=connectivity)
    
    return segm, convolved_data 

def get_segmentation(
    data,
    nsig,
    *,
    xpix=100,
    ypix=100,
    size=100,
    median_bkg=None,
    kernel_size=21,
    deblend=False,
    r_in=20,
    r_out=35,
    npixels=9,
    connectivity=8,
    invert=False,
    threshold=10,
    ):
    """
    Extract the segmentation map of a single central object in a postage stamp.

    A square cutout is taken around `(xpix, ypix)` (or the frame center if
    unspecified), background-subtracted, and segmented using `segm_find`.
    Central validation matches `morph_parameters` behavior:
      * If `threshold == 0`, require the exact central pixel to belong to a
        segmented object.
      * If `threshold > 0`, require that at least one segmented pixel lies
        within a central circular mask of radius `threshold`, then select the
        object whose centroid is **closest to the center** (from all segments).

    If validation fails, a zero array is returned.

    Parameters
    ----------
    data : ndarray
        2D image array.
    nsig : float
        Detection threshold in units of background sigma (passed to
        `segm_find`).
    xpix, ypix : int or None, optional
        Central pixel coordinates of the target object. If both are None,
        the image center is used. Default is 100 each.
    size : int, optional
        Side length (pixels) of the square cutout. If larger than the image
        dimensions, reduced to fit. Default is 100.
    median_bkg : float or None, optional
        Background estimate for the cutout. If None, a local annulus
        (radii `r_in`, `r_out`) is used. If 0, no subtraction is applied.
        Default is None.
    kernel_size : int, optional
        Size (pixels) of Gaussian kernel used in `segm_find`. Default is 21.
    deblend : bool, optional
        If True, deblend overlapping sources in the segmentation map.
        Default is False.
    r_in, r_out : int, optional
        Inner and outer radii (pixels) of the annulus used for local
        background estimation when `median_bkg` is None. Defaults are 20
        and 35.
    npixels : int, optional
        Minimum number of connected pixels above threshold required for
        detection (passed to `segm_find`). Default is 9.
    connectivity : int, optional
        Pixel connectivity: 4 (edge-connected) or 8 (edge+corner-connected).
        Default is 8.
    invert : bool, optional
        If True, swap (x, y) ordering when cropping. Default is False.
    threshold : int, optional
        Central validation parameter (pixels). See behavior above. Default is 10.

    Returns
    -------
    seg : ndarray
        Segmentation map of shape `(size, size)` with only the selected object
        retained (nonzero label). If validation fails, a zero array is returned.
    """

    # single-source use
    x0 = None if xpix is None else np.atleast_1d(xpix)[0]
    y0 = None if ypix is None else np.atleast_1d(ypix)[0]
    mbg = None if median_bkg is None else np.atleast_1d(median_bkg)[0]

    size = int(size)
    if size > min(data.shape):
        size = min(data.shape)

    # full-frame preview if user omitted coordinates
    if x0 is None and y0 is None:
        x0 = data.shape[1] // 2
        y0 = data.shape[0] // 2
        size = min(data.shape)

    # Crop and background-subtract
    stamp = data if size == data.shape[0] == data.shape[1] else data_processing.crop_image(
        data, int(x0), int(y0), size, invert=invert
    )

    if mbg is None:  # local background
        cx_ann = stamp.shape[1] / 2.0
        cy_ann = stamp.shape[0] / 2.0
        ann = CircularAnnulus((cx_ann, cy_ann), r_in=r_in, r_out=r_out)
        bg_stats = ApertureStats(stamp, ann, sigma_clip=SigmaClip())
        stamp = stamp - bg_stats.median
    elif mbg != 0:
        stamp = stamp - float(mbg)

    # Run segmentation
    segm, conv = segm_find(
        stamp,
        nsig=nsig,
        kernel_size=kernel_size,
        deblend=deblend,
        npixels=npixels,
        connectivity=connectivity,
    )

    if segm is None or getattr(segm, "data", None) is None:
        warn("No segmentation detected – returning zeros.", RuntimeWarning)
        return np.zeros((size, size))

    try:
        catalog = SourceCatalog(stamp, segm, convolved_data=conv)
    except Exception:
        warn("SourceCatalog failed – returning zeros.", RuntimeWarning)
        return np.zeros((size, size))

    # central validation to match `morph_parameters`
    h, w = segm.data.shape
    cx = w // 2
    cy = h // 2

    if int(threshold) == 0:
        # exact central-pixel membership required
        if segm.data[cy, cx] == 0:
            warn("No segment at exact center – returning zeros.", RuntimeWarning)
            return np.zeros((h, w))
        seg_label = segm.data[cy, cx]
        seg = np.array(segm.data, copy=True)
        seg[seg != seg_label] = 0
        return seg

    # threshold > 0: require at least one segmented pixel in central mask
    rr, cc = np.ogrid[:h, :w]
    mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= float(threshold) ** 2

    if np.count_nonzero(segm.data[mask]) == 0:
        warn("No segment in central mask – returning zeros.", RuntimeWarning)
        return np.zeros((h, w))

    labels_in_mask = np.unique(segm.data[mask])
    labels_in_mask = labels_in_mask[labels_in_mask != 0]   # drop background

    # map seg labels
    label_to_idx = {int(lbl): int(i) for i, lbl in enumerate(np.asarray(catalog.label))}
    idxs = [label_to_idx[int(lab)] for lab in labels_in_mask if int(lab) in label_to_idx]
    if not idxs:
        warn("No intersecting label found in catalog – returning zeros.", RuntimeWarning)
        return np.zeros((h, w))

    # pick the intersecting object whose centroid is closest to center
    dists = [np.hypot(float(catalog[i].centroid[0]) - cx, float(catalog[i].centroid[1]) - cy) for i in idxs]
    best_idx = idxs[int(np.argmin(dists))]
    best_label = catalog[best_idx].label

    # keep only the selected label
    seg = np.array(segm.data, copy=True)
    seg[seg != best_label] = 0

    return seg


def compute_layered_segmentation(
    image,
    sigma_values,
    pix_conversion,
    xpix,
    ypix,
    size,
    *,
    median_bkg=None,
    kernel_size=21,
    deblend=False,
    r_in=20,
    r_out=35,
    npixels=9,
    connectivity=8,
    threshold=10,
    ):
    """
    Generate a layered segmentation map across multiple detection thresholds.

    For each sigma threshold in `sigma_values`, `get_segmentation` is called to
    isolate the central source in a postage stamp. The resulting masks are stacked
    into a single 2D array, with different intensity values assigned to each layer.
    Higher sigma thresholds receive larger intensity values, creating a graded
    segmentation useful for visualization and contouring.

    Parameters
    ----------
    image : ndarray
        2D image array.
    sigma_values : sequence of float
        Detection thresholds (in sigma) to apply sequentially. Each value is
        passed as `nsig` to `get_segmentation`.
    pix_conversion : float
        Pixel-to-arcsecond (or other physical unit) conversion factor.
        Currently unused internally, but included for consistency with
        downstream plotting routines.
    xpix, ypix : int
        Pixel coordinates of the central source.
    size : int
        Side length (pixels) of the square cutout to extract.
    median_bkg : float or None, optional
        Background estimate for the cutout. If None, a local annulus
        (radii `r_in`, `r_out`) is used. If 0, no subtraction is applied.
        Default is None.
    kernel_size : int, optional
        Gaussian kernel size (pixels) used for segmentation convolution.
        Default is 21.
    deblend : bool, optional
        If True, apply deblending to split overlapping sources.
        Default is False.
    r_in, r_out : int, optional
        Inner and outer radii (pixels) for annular background estimation
        if `median_bkg` is None. Defaults are 20 and 35.
    npixels : int, optional
        Minimum number of connected pixels above threshold to define a
        source. Default is 9.
    connectivity : int, optional
        Pixel connectivity (4 or 8). Default is 8.
    threshold : int, optional
        Central circular mask radius (pixels) used in `get_segmentation`
        to enforce detection near the cutout center. Default is 10.

    Returns
    -------
    layered : ndarray of float, shape (size, size)
        Composite segmentation map. Pixels belonging to a detection at
        the i-th sigma level are assigned intensity values:
        [1.0, 0.7, 0.4, 0.1] (truncated to the number of sigma levels).
        Higher sigma levels correspond to higher intensities.

    Notes
    -----
    - If fewer than four sigma values are provided, only the corresponding
      fraction of the default intensity sequence is used.
    - If more than four sigma values are provided, only the first four are
      represented (later thresholds are ignored).
    - This function is designed primarily for visualization (e.g.,
      multi-level contours), not for quantitative feature extraction.
    """

    default_intensities = [1.0, 0.7, 0.4, 0.1]
    intensities = default_intensities[: len(sigma_values)]

    layered = np.zeros((size, size), dtype=float)

    for sval, inten in zip(sigma_values, intensities):
        segm = get_segmentation(
            data=image,
            nsig=sval,
            xpix=xpix,
            ypix=ypix,
            size=size,
            median_bkg=median_bkg,
            kernel_size=kernel_size,
            deblend=deblend,
            r_in=r_in,
            r_out=r_out,
            npixels=npixels,
            connectivity=connectivity,
            threshold=threshold,
        )
        layered[segm != 0] = inten

    return layered

def get_extent(img: np.ndarray, pix_conversion: float):
    """
    Compute the spatial extent of an image for `matplotlib.imshow`.

    The extent is centered on (0, 0) and converted from pixels to arcseconds
    (or other physical units) using the provided pixel-to-unit conversion.

    Parameters
    ----------
    img : ndarray
        2D image array.
    pix_conversion : float
        Pixel scale conversion factor (pixels per arcsecond, or pixels per unit).
        The returned extent is expressed in the reciprocal units (e.g., arcsec).

    Returns
    -------
    extent : list of float [xmin, xmax, ymin, ymax]
        Extent values suitable for passing to `imshow(extent=...)`, with
        coordinates centered on (0, 0).

    Notes
    -----
    - The extent is computed as:
      ``x = (arange(width) - width/2) / pix_conversion``,
      ``y = (arange(height) - height/2) / pix_conversion``.
    - Units depend on `pix_conversion`: for example, if
      `pix_conversion = 3.8961` pixels per arcsec, then the output extent
      is in arcseconds.
    """

    h, w = img.shape
    x = (np.arange(w) - w / 2) / pix_conversion
    y = (np.arange(h) - h / 2) / pix_conversion

    return [x.min(), x.max(), y.min(), y.max()]

def get_display_limits(img: np.ndarray):
    """
    Compute robust display limits for image visualization.

    The limits are based on the median and median absolute deviation (MAD) of
    finite pixels in the image, providing a stretch that is less sensitive to
    outliers than standard min/max scaling.

    Parameters
    ----------
    img : ndarray
        2D image array. Non-finite values (NaN, inf) are ignored.

    Returns
    -------
    vmin, vmax : float
        Lower and upper display limits, defined as:
        ``vmin = median - 3 * MAD``
        ``vmax = median + 10 * MAD``

    Notes
    -----
    - This stretch is useful for displaying faint, extended features while
      suppressing noise and extreme outliers.
    - MAD is defined as ``median(|x - median(x)|)``.
    - The asymmetric scaling (-3 × MAD, +10 × MAD) biases toward emphasizing
      positive flux features.
    """

    finite = np.isfinite(img)
    med = np.median(img[finite])
    mad = np.median(np.abs(img[finite] - med))
    
    return med - 3 * mad, med + 10 * mad

def plot_objects_segmentation(
    *images,
    pix_conversion=1.0,
    sigma_values=(0.1, 0.25, 0.55, 0.95),
    titles=None,
    suptitle="",
    xpix=None,
    ypix=None,
    size=None,
    median_bkg=None,
    kernel_size=21,
    deblend=False,
    r_in=20,
    r_out=35,
    npixels=9,
    connectivity=8,
    threshold=10,
    cmap="viridis",
    savepath="segm_multi.png",
    savefig=True,
    ):
    """
    Plot images alongside layered segmentation masks.

    Each input image is displayed in the top row, with its corresponding
    layered segmentation map (computed from `sigma_values`) displayed
    directly below. This provides a side-by-side visualization of raw
    data and threshold-dependent segmentation.

    Parameters
    ----------
    *images : ndarray
        One or more 2D image arrays. Between 1 and 5 are allowed.
    pix_conversion : float, optional
        Pixel-to-arcsecond (or other unit) conversion factor. Used to
        compute axis extents. Default is 1.0.
    sigma_values : sequence of float, optional
        Detection thresholds (in sigma) for layered segmentation.
        Default is (0.1, 0.25, 0.55, 0.95).
    titles : list of str or None, optional
        Titles for each image panel (top row). Length must match
        number of images. Default is no titles.
    suptitle : str, optional
        Figure-wide title. Default is "".
    xpix, ypix : int or None, optional
        Central pixel coordinates for cropping. If all of `xpix`, `ypix`,
        and `size` are given, each image is cropped to that region before
        plotting. Default is None.
    size : int or None, optional
        Side length (pixels) of cropped cutouts, if cropping is applied.
        Default is None (no cropping).
    median_bkg : float or None, optional
        Background estimate for segmentation. If None, background is
        estimated locally via annuli of radii `r_in` and `r_out`.
        If 0, no subtraction is applied. Default is None.
    kernel_size : int, optional
        Gaussian kernel size (pixels) for segmentation convolution.
        Default is 21.
    deblend : bool, optional
        If True, split overlapping sources during segmentation.
        Default is False.
    r_in, r_out : int, optional
        Inner and outer radii (pixels) for annular background estimation.
        Defaults are 20 and 35.
    npixels : int, optional
        Minimum connected pixel area (pixels) required for detection.
        Default is 9.
    connectivity : int, optional
        Pixel connectivity: 4 (edge-connected) or 8 (edge+corner-connected).
        Default is 8.
    threshold : int, optional
        Central circular mask radius (pixels) used to validate detections
        during segmentation. Default is 10.
    cmap : str, optional
        Matplotlib colormap for displaying the original images. Default
        is "viridis".
    savepath : str, optional
        Output path for saving the figure when `savefig=True`.
        Default is "segm_multi.png".
    savefig : bool, optional
        If True, save the figure to `savepath`. If False, display with
        `plt.show()`. Default is True.

    Returns
    -------
    None
        Displays or saves the figure. Does not return a DataFrame or array.

    Notes
    -----
    - A maximum of 5 images may be passed at once.
    - Each figure has two rows: raw images (top) and layered segmentation
      masks (bottom).
    - A legend indicates which intensity levels correspond to the
      `sigma_values` used in layered segmentation.
    - Axis labels are expressed in arcseconds if `pix_conversion` is
      given in pixels per arcsecond.
    """

    if not 1 <= len(images) <= 5:
        raise ValueError("Supply between 1 and 5 images.")

    ncols = len(images)
    titles = titles or [""] * ncols
    if len(titles) != ncols:
        raise ValueError("len(titles) must equal number of images.")

    proc_imgs, extents, vmins, vmaxs, layereds = [], [], [], [], []

    for img in images:
        if all(v is not None for v in (xpix, ypix, size)):
            img = data_processing.crop_image(img, xpix, ypix, size)

        extent = get_extent(img, pix_conversion)
        vmin, vmax = get_display_limits(img)
        layered = compute_layered_segmentation(
            img,
            sigma_values,
            pix_conversion,
            xpix=int(img.shape[1] / 2),
            ypix=int(img.shape[0] / 2),
            size=img.shape[0],
            median_bkg=median_bkg,
            kernel_size=kernel_size,
            deblend=deblend,
            r_in=r_in,
            r_out=r_out,
            npixels=npixels,
            connectivity=connectivity,
            threshold=threshold,
        )

        proc_imgs.append(img)
        extents.append(extent)
        vmins.append(vmin)
        vmaxs.append(vmax)
        layereds.append(layered)

    fig = plt.figure(figsize=(4 * ncols, 8))
    spec = gridspec.GridSpec(2, ncols, wspace=0, hspace=0)
    bin_cmap = plt.get_cmap("binary")

    for idx in range(ncols):

        ax = fig.add_subplot(spec[0, idx])
        ax.imshow(
            np.flip(proc_imgs[idx], axis=0),
            vmin=vmins[idx],
            vmax=vmaxs[idx],
            cmap=cmap,
            extent=extents[idx],
            origin="lower",
        )
        ax.set_title(titles[idx])
        ax.set_xticklabels([])
        if idx == 0:
            ax.set_ylabel(r"$\Delta \delta$ (arcsec)")
        else:
            ax.set_yticklabels([])

        ax_s = fig.add_subplot(spec[1, idx])
        ax_s.imshow(
            np.flip(layereds[idx], axis=0),
            cmap="binary",
            vmin=0,
            vmax=1,
            extent=extents[idx],
            origin="lower",
            interpolation="nearest",
        )
        if idx == 0:
            ax_s.set_ylabel(r"$\Delta \delta$ (arcsec)")
        else:
            ax_s.set_yticklabels([])
        ax_s.set_xlabel(r"$\Delta \alpha$ (arcsec)")

    fig.suptitle(suptitle, y=1.09)

    intensities = [1.0, 0.7, 0.4, 0.1][: len(sigma_values)]
    handles = [
        Patch(color=mcolors.to_hex(bin_cmap(i)), label=str(s))
        for i, s in zip(intensities, sigma_values)
    ]

    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        handlelength=1.0,
        title=r"$\sigma_{\rm det}$",
        bbox_to_anchor=(0.5, 1.063),
        ncol=len(sigma_values),
        frameon=True,
        fancybox=True
    )

    if savefig:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_images_grid_2x2(
    img1, img2, img3, img4,
    *,
    pix_conversion=1.0,
    titles=None,
    suptitle="",
    xpix=None,
    ypix=None,
    size=None,
    cmap="viridis",
    savepath="outliers.png",
    savefig=True,
    ):
    """
    Display four images in a 2×2 grid with consistent visualization settings.

    This function mirrors the style of the *image panels* from
    `plot_objects_segmentation`: identical extent handling, robust
    (median±MAD) display limits, axis labeling, spacing, and aesthetics.
    Useful for side-by-side comparison of four sources or cutouts.

    Parameters
    ----------
    img1, img2, img3, img4 : ndarray
        Four 2D image arrays to display.
    pix_conversion : float, optional
        Pixel-to-arcsecond (or other unit) conversion factor. Passed to
        `get_extent`. Default is 1.0.
    titles : list of str or None, optional
        Titles for the four panels, in row-major order
        ([top-left, top-right, bottom-left, bottom-right]).
        Must be length 4. Default is None (no titles).
    suptitle : str, optional
        Figure-level title. Default is "".
    xpix, ypix, size : int or None, optional
        If all three are provided, each image is cropped to a square
        cutout using `data_processing.crop_image(img, xpix, ypix, size)`.
        Default is None (no cropping).
    cmap : str, optional
        Matplotlib colormap for images. Default is "viridis".
    savepath : str, optional
        Output file path if saving the figure. Default is "outliers.png".
    savefig : bool, optional
        If True, save the figure to `savepath`. If False, display the
        figure interactively with `plt.show()`. Default is True.

    Returns
    -------
    None
        Displays or saves the figure. Does not return arrays or DataFrames.

    Notes
    -----
    - Each panel is flipped vertically (consistent with other plotting
      functions in this module) and labeled in arcseconds relative to
      the cutout center.
    - Robust display scaling is applied via `get_display_limits`.
    - Grid layout uses equal spacing (no white space between panels).
    """

    images = [img1, img2, img3, img4]

    titles = (titles or ["", "", "", ""])
    if len(titles) != 4:
        raise ValueError("titles must be a list of four strings (row-major order).")

    proc, extents, vmins, vmaxs = [], [], [], []
    for img in images:
        im = img
        if all(v is not None for v in (xpix, ypix, size)):
            im = data_processing.crop_image(im, xpix, ypix, size)

        extent = get_extent(im, pix_conversion)
        vmin, vmax = get_display_limits(im)

        proc.append(im)
        extents.append(extent)
        vmins.append(vmin)
        vmaxs.append(vmax)

    fig = plt.figure(figsize=(8, 8))
    spec = gridspec.GridSpec(2, 2, wspace=0, hspace=0)

    # Helper to set axes the same way as in the original image panels
    def _style_ax(ax, row, col):
        # Titles on top row panels
        ax.set_title(titles[row * 2 + col])

        # Remove x tick labels on the top row
        if row == 0:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\Delta \alpha$ (arcsec)")

        # Left column keeps y labels; right column removes them
        if col == 0:
            ax.set_ylabel(r"$\Delta \delta$ (arcsec)")
        else:
            ax.set_yticklabels([])

    # Draw the four panels
    for row in range(2):
        for col in range(2):
            idx = row * 2 + col
            ax = fig.add_subplot(spec[row, col])
            ax.imshow(
                np.flip(proc[idx], axis=0),
                vmin=vmins[idx],
                vmax=vmaxs[idx],
                cmap=cmap,
                extent=extents[idx],
                origin="lower",
            )
            _style_ax(ax, row, col)

    fig.suptitle(suptitle, y=0.935)

    if savefig:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def align_error_array(data, error, data_coords, error_coords):
    """
    Align an error map with a data array by shifting and padding.

    This function shifts the error array so that a reference object
    (identified by coordinates in both arrays) is aligned, then pads
    or crops as needed to match the shape of `data`. Useful when
    error maps (e.g., rms images) are offset relative to science
    images due to inconsistent sizes or coordinate origins.

    Parameters
    ----------
    data : ndarray
        Target 2D science image array.
    error : ndarray
        Error map (e.g., rms image) to align with `data`.
    data_coords : tuple of int
        (x, y) pixel coordinates of a reference object in `data`.
    error_coords : tuple of int
        (x, y) pixel coordinates of the same object in `error`.

    Returns
    -------
    padded_error : ndarray
        Error array aligned and padded/cropped to the same shape
        as `data`. Non-overlapping regions are filled with zeros.

    Notes
    -----
    - This approach assumes both arrays are on the same pixel grid
      up to an integer shift, and does not perform interpolation.
    - Alignment is determined by the relative offset between
      `data_coords` and `error_coords`.
    - This was originally motivated by the NDWFS Boötes R-band
      data, where rms maps and images had inconsistent dimensions.
      In general, a WCS-based solution (via `astropy.wcs`) may be
      preferable if RA/Dec information is available.
    """

    #Calculate the required shifts in x and y directions
    x_shift, y_shift = data_coords[0] - error_coords[0], data_coords[1] - error_coords[1]

    #Pad the error array with zeros to match the data array size
    padded_error = np.zeros_like(data)

    #Determine the start and end indices for the error array
    error_start_x, error_end_x = max(0, -x_shift), min(error.shape[1], data.shape[1] - x_shift)
    error_start_y, error_end_y = max(0, -y_shift), min(error.shape[0], data.shape[0] - y_shift)
    
    #Determine the start and end indices for the data array
    data_start_x, data_end_x = max(0, x_shift), min(data.shape[1], error.shape[1] + x_shift)
    data_start_y, data_end_y = max(0, y_shift), min(data.shape[0], error.shape[0] + y_shift)
    
    #Copy the relevant portion of the error array to the padded_error array
    padded_error[data_start_y:data_end_y, data_start_x:data_end_x] = error[error_start_y:error_end_y, error_start_x:error_end_x]
    
    return padded_error


