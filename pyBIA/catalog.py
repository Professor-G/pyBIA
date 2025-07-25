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
    Photometric + morphological catalog builder for postage-stamp images.
    """

    #
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
        nsig: float = 0.7,
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

        # 
        self.bkg = bkg
        self.cat = cat

        # 
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

    def create(self, *, save_file: bool = True, path: str | None = None,
               filename: str | None = None):
        """
        Build full catalog; returns a `pandas.DataFrame`.

        Parameters
        ----------
        save_file : bool
            Save CSV output to disk.
        path : str or None
            Target directory (defaults to `Path.home()`).
        filename : str or None
            Output filename (defaults to ``pyBIA_catalog.csv``).
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

    # 
    def _auto_detect_sources(self):
        """Run segmentation on whole frame and build catalog."""

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
                median_bkg=None, invert=self.invert, deblend=self.deblend
            )
            tbl = make_table(props_list, moments)
        else:
            tbl = None

        self.cat = make_dataframe(
            table=tbl, x=self.x, y=self.y, zp=self.zp,
            obj_name=self.obj_name, field_name=self.field_name, flag=self.flag,
            flux=aper_stats.sum, flux_err=flux_err, median_bkg=None
        )

    # 
    def _aperture_photometry(self):
        """Photometry for user-supplied positions."""

        positions = list(zip(self.x, self.y))
        apertures = CircularAperture(positions, r=self.aperture)
        aper_stats = ApertureStats(self.data, apertures, error=self.error)

        # local background per source
        if self.bkg is None:
            ann = CircularAnnulus(positions, r_in=self.annulus_in, r_out=self.annulus_out)
            bkg_stats = ApertureStats(self.data, ann, error=self.error, sigma_clip=SigmaClip())
            bkg = bkg_stats.median
            flux = aper_stats.sum - bkg * apertures.area
        else: # data already backgroud-subtracted 
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

    # Only used when no catalog positions are input 
    def _subtract_global_background(self):
        """Estimate median sky in 2×annulus_out square per object."""

        length = self.annulus_out * 2 * 2 #The sub-array when padding will be a square encapsulating the outer annuli
        if (self.data.shape[0] < length) or (self.data.shape[1] < length):
            bg = sigma_clipped_stats(self.data)[1]
            return self.data - bg
        return subtract_background(self.data, length=length)



def morph_parameters(data, x, y, size=100, nsig=0.6, threshold=10, kernel_size=21, npixels=9, connectivity=8, median_bkg=None, 
    invert=False, deblend=False, exptime=None):
    """
    Applies image segmentation on each object to calculate morphological 
    parameters calculated from the moment-based properties. These parameters 
    can be used to train a machine learning classifier.

    By default the data is assumed to be background subtracted, otherwise the
    median_bkg argument needs to be set for proper image detection.
    
    Args:
        data (ndarray): 2D array.
        x (ndarray): 1D array or list containing the x-pixel position.
            Can contain one position or multiple samples.
        y (ndarray): 1D array or list containing the y-pixel position.
            Can contain one position or multiple samples.
        size (int, optional): The size of the box to consider when calculating
            features related to the local environment. Default is 100x100 pixels.
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.6.
        threshold (int): To avoid non-detections the segmentation map will be cropped at the center,
            this central subarray will be of size (threshold x threshold). If no segmentation object
            is located in this central area, the source will be flagged as a non-detection. Defaults
            to 5 pixels.
        median_bkg (ndarray, optional): 1D array containing the median background
            in the annuli of each around each (x,y) object. This is not a standard rms or background
            map input. Defaults to None, in which case data is assumed to be background-subtracted.
        invert (bool): If True the x & y coordinates will be switched
            when cropping out the object, see Note below. Defaults to False.
        deblend (bool, optional): If True, the objects are deblended during the segmentation
            procedure, thus deblending the objects before the morphological features
            are computed. Defaults to False so as to keep blobs as one segmentation object.
        exptime (float, optional):

    Note:
        This function requires x & y positions as each source 
        is isolated before the image segmentation is performed as this is
        computationally more efficient. If you need the x and y positions, you can
        run the catalog.create function, which will include the x & y pixel 
        positions of all cataloged sources.

        IMPORTANT: When loading data from a .fits file the pixel convention
        is switched. The (x, y) = (0, 0) position is on the top left corner of the .fits
        image. The standard convention is for the (x, y) = (0, 0) to be at the bottom left
        corner of the data. We strongly recommend you double-check your data coordinate
        convention. We made use of .fits data with the (x, y) = (0, 0) position at the top
        left of the image, for this reason we switched x and y when cropping out individual
        objects. The parameter invert=True performs the coordinate switch for us. This is only
        required because pyBIA's cropping function assumes standard convention.
    
    Return:
        A catalog of morphological parameters. If multiple positions are input, then the
        output will be a list containing multiple morphological catalogs, one for
        each position.
        
    """

    if data.shape[0] < 100:
        print('Small image warning: results may be unstable if the object does not fit entirely within the frame.')
    try: #If position array is a single number it will be converted into a list of unit length
        __ = len(x)
    except:
        x, y = [x], [y]

    size = size if data.shape[0] > size and data.shape[1] > size else min(data.shape[0],data.shape[1])

    prop_list, moment_list = [], []
    progess_bar = bar.FillingSquaresBar('Applying image segmentation...', max=len(x))

    for i in range(len(x)):
        new_data = data_processing.crop_image(data, int(x[i]), int(y[i]), size, invert=invert)
        if median_bkg is not None:
            new_data -= median_bkg[i] 
        if exptime is not None:
            new_data /= exptime
       
        segm, convolved_data = segm_find(new_data, nsig=nsig, kernel_size=kernel_size, deblend=deblend, npixels=npixels, connectivity=connectivity)
        try:
            props = SourceCatalog(new_data, segm, convolved_data=convolved_data)
        except:
            prop_list.append(-999), moment_list.append(-999) #If there are no segmented objects in the image
            progess_bar.next()
            continue

        # Mask a circular area at the center of the image, using radius=threshold
        # Flag if there is no segmented object within the circular mask 
        cx = cy = int(size / 2)
        x_pos, y_pos = np.ogrid[:new_data.shape[0], :new_data.shape[1]]
        r2 = (x_pos - cx) ** 2 + (y_pos - cy) ** 2
        mask = r2 <= threshold ** 2

        if np.count_nonzero(segm.data[mask]) == 0: 
            prop_list.append(-999), moment_list.append(-999)
            progess_bar.next()
            continue

        sep_list=[]
        for xx in range(len(props)): #This is to select the segmented object closest to the center, (x,y)=(size/2, size/2)
            xcen = float(props[xx].centroid[0])
            ycen = float(props[xx].centroid[1])
            sep_list.append(np.sqrt((xcen-(size/2))**2 + (ycen-(size/2))**2))

        inx = np.where(sep_list == np.min(sep_list))[0]
        if len(inx) > 1: #In case objects can't be deblended
            inx = inx[0] 

        ##### Image Moments #####
        new_data[segm.data != props[inx].label] = 0
        moments_table = make_moments_table(new_data)
        
        prop_list.append(props[inx]), moment_list.append(moments_table)
        progess_bar.next()
    progess_bar.finish()

    if len(prop_list) != len(moment_list):
        raise ValueError('The properties list does not match the image moments list.')
    
    if -999 in prop_list:
        print('NOTE: At least one object could not be detected in segmentation, perhaps the object is too faint. The morphological features have been set to -999.')
        return np.array(prop_list, dtype=object), moment_list, np.zeros(new_data.shape)
    else:
        return np.array(prop_list, dtype=object), moment_list, segm.data

def make_table(props, moments):
    """
    Returns the morphological parameters calculated from the sementation image.
    A list of the parameters and their function is available in the Photutils
    Source Catalog documentation: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html
    
    Args:
        Props (source catalog): A source catalog containing the segmentation parameters.
        
    Returns:
        Array containing the morphological features. 

    """
    moment_list = [
        'M00', 'M10', 'M01', 'M20', 'M11', 'M02', 'M30', 'M21', 'M12', 'M03',
        'mu00', 'mu10', 'mu01', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
        'G00', 'G10', 'G01', 'G20', 'G11', 'G02', 'G30', 'G21', 'G12', 'G03',
        'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7',
        'L00', 'L10', 'L01', 'L20', 'L11', 'L02', 'L30', 'L21', 'L12', 'L03'
    ]

    prop_list = ['area', 'covar_sigx2', 'covar_sigy2', 'covar_sigxy', 'covariance_eigvals', 
        'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 
        'fwhm', 'gini', 'orientation', 'perimeter', 'semimajor_sigma', 'semiminor_sigma',
        'isscalar', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'max_value', 'maxval_xindex', 
        'maxval_yindex', 'min_value', 'minval_xindex', 'minval_yindex', 'moments', 'moments_central']
    
    table = []
    print('Writing catalog...')
    for i in range(len(props)):

        morph_feats = []

        try:
            props[i][0].area #To avoid when this is None
            for moment in moment_list:
                morph_feats.append(float(moments[i][moment]))
        except:
            for j in range(len(prop_list+moment_list)+31): #+1 because covariance eigenvalue param is actually 2 params, and +30 for the 2 4x4 moment matrices 
                morph_feats.append(-999)
            table.append(morph_feats)
            continue

        QTable = props[i][0].to_table(columns=prop_list)
        for param in prop_list:
            if param == 'moments' or param == 'moments_central': #To 3rd order photutils outputs a 4x4 matrix (obselete?)
                for moment in np.ravel(QTable[param]):
                    morph_feats.append(moment)
            elif param == 'covariance_eigvals': 
                morph_feats.append(np.ravel(QTable[param])[1].value)
                morph_feats.append(np.ravel(QTable[param])[0].value) #This is the second eigval
            elif param == 'isscalar':
                if QTable[param] == True: #Checks whether it's a single source, 1 for true, 0 for false
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)
            elif param == 'bbox': #Calculate area of bounding box
                morph_feats.append(props[i][0].bbox.shape[0] * props[i][0].bbox.shape[1])
            else:
                morph_feats.append(QTable[param].value[0])

        table.append(morph_feats)

    return np.array(table, dtype=object)

def make_dataframe(table=None, x=None, y=None, zp=None, flux=None, flux_err=None, median_bkg=None, 
    obj_name=None, field_name=None, flag=None, save=True, path=None, filename=None):
    """
    This function takes as input the catalog of morphological features
    and other metrics and compiles the data as a Pandas dataframe. 

    Args:
        table (ndarray, optional): Array containing the object features. Can make with make_table() function.
            If None then a Pandas dataframe containing only the input columns will be generated. Defaults to None.
        x (ndarray, optional): 1D array containing the x-pixel position.
            If input it must be an array of x positions for all objects in the table. 
            This x position will be appended to the dataframe for cataloging purposes. Defaults to None.
        y (ndarray, optional): 1D array containing the y-pixel position.
            If input it must be an array of y positions for all objects in the table. 
            This y position will be appended to the dataframe for cataloging purposes. Defaults to None.
        zp (float): Zeropoint of the instrument.
        exptime (float): Not currently used.
        flux (ndarray, optional): 1D array containing the calculated flux
            of each object. This will be appended to the dataframe for cataloging purposes. Defaults to None.
        flux_err (ndarray, optional): 1D array containing the calculated flux error
            of each object. This will be appended to the dataframe for cataloging purposes. Defaults to None.
        median_bkg (ndarray, optional):  1D array containing the median background around the source annuli.
            This will be appended to the dataframe for cataloging purposes. Defaults to None.
        name (ndarray, str, optional): A corresponding array or list of object name(s). This will be appended to 
            the dataframe for cataloging purposes. Defaults to None.
        flag (ndarray, optional): 1D array containing a flag value for each object corresponding
            to the x & y positions. Defaults to None. 
        save (bool, optional): If False the dataframe CSV file will not be saved to the local
            directory. Defaults to True. 
        path (str, optional): Absolute path where CSV file should be saved, if save=True. If 
            path is not set, the file will be saved to the local directory.
        filename(str, optional): Name of the output catalog. Default name is 'pyBIA_catalog'.

    Note:
        These features can be used to create a machine learning model. 

    Example:

        >>> props, moments = morph_parameters(data, x=xpix, y=ypix)
        >>> table = make_table(props, moments)
        >>> dataframe = make_dataframe(table, x=xpix, y=ypix)

    Returns:
        Pandas dataframe containing the parameters and features of all objects
        in the input data table. If save=True, a CSV file titled 'pybia_catalog'
        will be saved to the local directory, unless a path is specified.

    """

    if filename is None:
        filename = 'pyBIA_catalog'

    prop_list = [
        'M00', 'M10', 'M01', 'M20', 'M11', 'M02', 'M30', 'M21', 'M12', 'M03',
        'mu00', 'mu10', 'mu01', 'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
        'G00', 'G10', 'G01', 'G20', 'G11', 'G02', 'G30', 'G21', 'G12', 'G03',
        'Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7',
        'L00', 'L10', 'L01', 'L20', 'L11', 'L02', 'L30', 'L21', 'L12', 'L03', 
        'area', 'covar_sigx2', 'covar_sigy2', 'covar_sigxy', 'covariance_eigval1', 'covariance_eigval2', 
        'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 
        'fwhm', 'gini', 'orientation', 'perimeter', 'semimajor_sigma', 'semiminor_sigma',
        'isscalar', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'max_value', 'maxval_xindex', 
        'maxval_yindex', 'min_value', 'minval_xindex', 'minval_yindex'
        ]

    for i in range(16): #Photutils API returns 4x4 matrix
        prop_list = prop_list + ['moments_'+str(i)]
    for i in range(16):
        prop_list = prop_list + ['moments_central_'+str(i)]

    data_dict = {}

    if obj_name is not None:
        data_dict['obj_name'] = obj_name
    if field_name is not None:
        data_dict['field_name'] = field_name
    if flag is not None:
        data_dict['flag'] = flag
    if x is not None:
        data_dict['xpix'] = x
    if y is not None:
        data_dict['ypix'] = y
    if median_bkg is not None:
        data_dict['median_bkg'] = median_bkg
    if flux is not None:
        if zp is None:
            data_dict['flux'] = flux
        else:
            data_dict['flux'] = flux
            data_dict['mag'] = -2.5*np.log10(np.array(flux))+zp 
    if flux_err is not None:
        if zp is None:
            data_dict['flux_err'] = flux_err
        else:
            data_dict['flux_err'] = flux_err
            data_dict['mag_err'] = (2.5/np.log(10))*(np.array(flux_err)/np.array(flux))
    
    if table is None:
        df = pd.DataFrame(data_dict)
        if save == True:
            if path is None:
                print("No path specified, saving catalog to local home directory.")
                path = str(Path.home())+'/'
            df.to_csv(path+filename, index=False) 
            return df
        return df

    try:
        __ = len(table)
    except: #TypeError
        table = [table]

    for i in range(len(prop_list)):
        data_dict[prop_list[i]] = table[:,i]

    df = pd.DataFrame(data_dict)
    if save == True:
        if path is None:
            print("No path specified, saving catalog to local home directory.")
            path = str(Path.home())+'/'
        df.to_csv(path+filename, index=False) 
        return df
    return df    

def subtract_background(data, length=150):
    """
    Removes the background by subtracting the local median pixel value 
    in sub-regions of size (length x length). The data matrix will be 
    padded accordingly usying symmetrical boundary conditions to ensure
    the local regions can expand evenly.

    Args:
        data (ndarray): 2D array of a single image.
        length (int): The length of the rectangular local regions. Default
            is 150 pixels, thus the local background is subtracted by calculating
            a robust median in 150x150 regions.

    Returns:
        The background subtracted data array.
    """

    Nx, Ny = data.shape[1], data.shape[0]
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

def segm_find(data: np.ndarray, *, nsig: float = 0.6, kernel_size: int = 21, deblend: bool = False, npixels: int = 9, connectivity: int = 8):
    """
    Finds objects using the segmentation detection threshold. 
    
    Note:
        Data must be background subtracted.

    Args:
        data (ndarray): 2D array of a single image.
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.6.
        kernel_size (int): The size lenght of the square Gaussian filter kernel used to convolve 
            the data. This length must be odd. Defaults to 21.
        deblend (bool, optional): If True, the objects are deblended during the segmentation
            procedure, thus deblending the objects before the morphological features
            are computed. Defaults to False so as to keep blobs as one segmentation object.
        npxiels (int): From photutils: Detected sources must have npixels connected pixels that are each greater than the threshold value in the input data
        connectivity (int): From photutils: The type of pixel connectivity used in determining how pixels are grouped into a detected source. The options are 4 or 8 (default). 4-connected pixels touch along their edges. 8-connected pixels touch along their edges or corners.
    Returns:
        First output is the segmentation image object, the second output is the convolved data
        that was used when cataloging the segmentation objects.

    """
    threshold = detect_threshold(data, nsigma=nsig, background=0.0)
    sigma_pix = 9.0 * gaussian_fwhm_to_sigma   # FWHM = 9. smooth the data with a 2D circular Gaussian kernel with a FWHM of 3 pixels to filter the image prior to thresholding:
    kernel = Gaussian2DKernel(sigma_pix, x_size=kernel_size, y_size=kernel_size, mode='center')
    convolved_data = convolve(data, kernel, normalize_kernel=True, preserve_nan=True)
    segm = detect_sources(convolved_data, threshold, npixels=npixels, connectivity=connectivity)
    if deblend and segm is not None:
        segm = deblend_sources(convolved_data, segm, npixels=npixels, connectivity=connectivity)
    
    return segm, convolved_data 

def get_segmentation(data, nsig, pix_conversion, xpix=100, ypix=100, size=100, median_bkg=0, kernel_size=21, deblend=False, r_in=20, r_out=35, npixels=9, connectivity=8, invert=False, threshold=10):
    """
    INTENDED TO WORK ON AN IMAGE WHERE ONLY ONE OBJECT IS OF INTEREST! THUS IN PRINCIPLE WE MUST ENSURE XPIX AND YPIX TAKE AT MOST ONE VALUE
    """

    if data.shape[1] < size:
        size = data.shape[1]

    if xpix is None and ypix is None:
        xpix, ypix = data.shape[1]/2, data.shape[1]/2
        size = data.shape[1]

    try: 
        __ = len(xpix)
    except:
        xpix = [xpix]
    try:
        __ = len(ypix)
    except:
        ypix = [ypix]
    try:
        __ = len(median_bkg)
    except:
        if median_bkg is not None:
            median_bkg = [median_bkg]
        
    for i in range(len(xpix)):
        if size == data.shape[1]:
            new_data = data
        else: 
            new_data = data_processing.crop_image(data, int(xpix[i]), int(ypix[i]), size, invert=invert)

        if median_bkg is None: #Hard coding annuli size, inner:25 -> outer:35
            print("Subtracting background...")
            if new_data.shape[0] > 200 and len(xpix) == 1:
                print('Calculating background in local regions, this will take a while... if data is background subtracted set median_bkg=0.')
                new_data = subtract_background(new_data)
            else:
                annulus_apertures = CircularAnnulus((new_data.shape[1]/2, new_data.shape[0]/2), r_in=r_in, r_out=r_out)
                bkg_stats = ApertureStats(new_data, annulus_apertures, sigma_clip=SigmaClip())
                new_data -= bkg_stats.median
        elif median_bkg == 0:
            new_data -= median_bkg 
        else:
            new_data -= median_bkg[i]

        segm, convolved_data = segm_find(new_data, nsig=nsig, kernel_size=kernel_size, deblend=deblend, npixels=npixels, connectivity=connectivity)

        try:
            _ = segm.data
        except AttributeError:
            print(f"DETECTION WARNING: No segmentation patches could be generated anywhere on the image for sigma={nsig}, kernel size={kernel_size}, npixels={npixels}, and connectivity={connectivity}. Adjust the detection settings and try again! Returning zero-like array...")
            segm = np.zeros((size, size))
            return segm

        try:
            props = SourceCatalog(new_data, segm, convolved_data=convolved_data)
        except:
            print(f"CATALOG WARNING: No source catalog could be generated for sigma={nsig}, kernel size={kernel_size}, npixels={npixels}, and connectivity={connectivity}. Adjust the detection settings and try again! Returning zero-like array...")
            segm = np.zeros((size, size))
            return segm

        # Mask a circular area at the center of the image, using radius=threshold
        # Flag if there is no segmented object within the circular mask 
        cx = cy = int(size / 2)
        x_pos, y_pos = np.ogrid[:new_data.shape[0], :new_data.shape[1]]
        r2 = (x_pos - cx) ** 2 + (y_pos - cy) ** 2
        mask = r2 <= threshold ** 2

        if np.count_nonzero(segm.data[mask]) == 0:
            print(f"DETECTION WARNING: No segmentation patches present within a circular mask of radius (threshold)={threshold}, for sigma={nsig}, kernel size={kernel_size}, npixels={npixels}, and connectivity={connectivity}. Returning zero-like array...")
            segm = np.zeros((size, size))
            return segm

        sep_list=[]
        for xx in range(len(props)): #This is to select the segmented object closest to the center, (x,y)=(size/2, size/2)
            xcen = float(props[xx].centroid[0])
            ycen = float(props[xx].centroid[1])
            sep_list.append(np.sqrt((xcen-(size/2))**2 + (ycen-(size/2))**2))

        inx = np.where(sep_list == np.min(sep_list))[0]
        if len(inx) > 1:
            inx = inx[0] 

        segm.data[segm.data != props[inx].label] = 0

    return segm.data


def compute_layered_segmentation(image, sigma_values, pix_conversion, xpix, ypix, size, median_bkg=0, kernel_size=21, deblend=False, r_in=20, r_out=35, npixels=9, connectivity=8, threshold=10):
    """
    For each sigma threshold in sigma_values, compute a segmentation mask (by calling
    get_segmentation with data1 and data2 – here if only one image is available we pass
    the same image twice). Each mask is then normalized and stored.
    
    Finally, a layered segmentation image is built by overlaying the masks.
    (The first threshold’s mask is assigned intensity 1.0, the second 0.7, then 0.4 and 0.1.)
    """
    segm_list = []
    # Use the default intensities for up to four sigma values:
    default_intensities = [1.0, 0.7, 0.4, 0.1]
    #default_intensities = np.linspace(0.1, 1, 4) ** 0.5  # sqrt for better perceptual contrast
    #default_intensities = default_intensities[::-1]
    intensities = default_intensities[:len(sigma_values)]
    
    for sval in sigma_values:
        segm = get_segmentation(data=image, nsig=sval, pix_conversion=pix_conversion, xpix=xpix, ypix=ypix, size=size, median_bkg=median_bkg, kernel_size=kernel_size, deblend=deblend, r_in=r_in, r_out=r_out, npixels=npixels, connectivity=connectivity, threshold=threshold)
        segm[segm != 0] = 1000
        segm_list.append(segm)
    
    # Build the layered segmentation image
    layered = np.zeros_like(segm_list[0], dtype=float)
    for segm, inten in zip(segm_list, intensities):
        layered[segm == 1000] = inten
    return layered

def get_extent(img, pix_conversion):
    """
    Returns an extent for imshow in arcsec assuming the image is centered.
    """
    height, width = img.shape
    center_x = width // 2
    center_y = height // 2
    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    x_arcsec = x / pix_conversion
    y_arcsec = y / pix_conversion
    return [x_arcsec.min(), x_arcsec.max(), y_arcsec.min(), y_arcsec.max()]

def get_display_limits(img):
    """
    Compute robust display limits based on the median and robust std (using median absolute deviation).
    """
    finite = np.isfinite(img)
    med = np.median(img[finite])
    std = np.median(np.abs(img[finite] - med))
    return med - 3*std, med + 10*std


def plot_objects_segmentation(
        *images,
        pix_conversion=1.0,
        sigma_values=[0.1, 0.25, 0.55, 0.95],
        titles=None,
        suptitle='',
        xpix=None,
        ypix=None,
        size=None,
        median_bkg=0, kernel_size=21, deblend=False, r_in=20, r_out=35, npixels=9, connectivity=8, threshold=10, cmap='viridis',
        savepath='/Users/daniel/Desktop/segm_multi.png', savefig=True):
    """
    Plot up to five objects (default behaviour still works for one or two).

    Parameters
    ----------
    *images : 2-D numpy arrays
        One to five postage-stamp images.  The first row shows the data,
        the second the layered segmentation masks.
    pix_conversion : float
        Pixels-to-arcsec conversion factor.
    sigma_values : list
        Detection‐σ thresholds (≤ 4 values), highest σ plotted brightest.
    titles : list/tuple of str or None
        Per-panel titles for the imaging row; supply len(titles) == len(images).
    suptitle : str
        Global figure title.
    xpix, ypix, size : int or None
        If all three are given each image is cropped with
        ``crop_image(img, xpix, ypix, size)`` before processing.
    savepath : str
        Path to ``plt.savefig`` output.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not 1 <= len(images) <= 5:
        raise ValueError('Supply between 1 and 5 images.')

    ncols = len(images)
    if titles is None:
        titles = [''] * ncols
    elif len(titles) != ncols:
        raise ValueError('len(titles) must equal number of images.')

    proc_imgs, extents, vmins, vmaxs, layereds = [], [], [], [], []

    # ------------------------------------------------------------------
    for img in images:
        # optional crop
        if (xpix is not None) and (ypix is not None) and (size is not None):
            img = data_processing.crop_image(img, xpix, ypix, size)

        extent = get_extent(img, pix_conversion)
        vmin, vmax = get_display_limits(img)
        layered = compute_layered_segmentation(
            img, sigma_values, pix_conversion,
            xpix=int(img.shape[0]/2),#xpix if xpix is not None else img.shape[0]/2,
            ypix=int(img.shape[1]/2),#ypix if ypix is not None else img.shape[1]/2,
            size=int(img.shape[0]), #size if size is not None else img.shape[0],
            median_bkg=median_bkg, kernel_size=kernel_size, deblend=deblend, r_in=r_in, r_out=r_out, npixels=npixels, connectivity=connectivity, threshold=threshold
            )

        proc_imgs.append(img)
        extents.append(extent)
        vmins.append(vmin)
        vmaxs.append(vmax)
        layereds.append(layered)

    fig_w = 4 * ncols          # keep aspect similar to original (8×8 for 2 cols)
    fig = plt.figure(figsize=(fig_w, 8))
    spec = gridspec.GridSpec(2, ncols, wspace=0, hspace=0)

    binary_cmap = plt.get_cmap('binary')

    for idx in range(ncols):
        # --- imaging row ---
        ax_img = fig.add_subplot(spec[0, idx])
        ax_img.imshow(np.flip(proc_imgs[idx], axis=0),
                      vmin=vmins[idx], vmax=vmaxs[idx],
                      cmap=cmap, extent=extents[idx], origin='lower')
        ax_img.set_title(f'{titles[idx]}')
        if idx != 0:
            ax_img.set_yticklabels([])
        else:
            ax_img.set_ylabel(r'$\Delta \delta$ (arcsec)')
        ax_img.set_xticklabels([])

        ax_seg = fig.add_subplot(spec[1, idx])
        ax_seg.imshow(np.flip(layereds[idx], axis=0),
                      cmap='binary', vmin=0, vmax=1,
                      extent=extents[idx], origin='lower', interpolation='nearest')
        if idx == 0:
            ax_seg.set_ylabel(r'$\Delta \delta$ (arcsec)')
        else:
            ax_seg.set_yticklabels([])
        ax_seg.set_xlabel(r'$\Delta \alpha$ (arcsec)')

    fig.suptitle(suptitle, y=1.09)

    default_intensities = [1.0, 0.7, 0.4, 0.1]
    intensities_used = default_intensities[:len(sigma_values)]
    legend_handles = [
        Patch(color=mcolors.to_hex(binary_cmap(inten)),
              label=f'{sigma}') for sigma, inten in zip(sigma_values, intensities_used)
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc='upper center', handlelength=1.,
               title=r'$\sigma_{\rm det}$',
               bbox_to_anchor=(0.5, 1.063), ncol=len(sigma_values),
               frameon=True, fancybox=True)

    if savefig:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def align_error_array(data, error, data_coords, error_coords):
    """
    Aligns the error array with the data array by shifting and padding the error array.
    This can be used in the event that the error map size does not match the data size.
    By manually identifying the coordinates of a prominent object in both arrays,
    this function can be used to perform the proper alignment and padding/cropping. 
    This was used as the NDWFS Bootes R-band data size was inconsistent with the corresponding rms maps,
    causing the pixel locations to be inconsistent, although this can be worked around by using
    the RA and DEC and invoking the WCS from the astropy API.

    Args:
        data (ndarray): The data array.
        error (ndarray): The error array.
        data_coords (tuple): The (x, y) coordinates of an object in the data array. Must be integers.
        error_coords (tuple): The (x, y) coordinates of the same object in the error array. Must be integers.

    Returns:
        ndarray: The aligned and padded error array.
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
