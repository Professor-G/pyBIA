# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:10:11 2021

@author: danielgodinez
"""
from astropy.utils.exceptions import AstropyWarning
from warnings import warn, filterwarnings, simplefilter
filterwarnings("ignore", category=AstropyWarning) #Ignore NaN & inf warnings
filterwarnings("ignore", category=RuntimeWarning) #Ignore sigma_clipping NaN warning
from astropy.io import fits 
from astropy.wcs import WCS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.patches import Patch
import matplotlib.colors
from matplotlib import gridspec
#import matplotlib

from pyBIA.data_processing import crop_image


from cycler import cycler 

from photutils.detection import DAOStarFinder
from photutils import detect_threshold, detect_sources, deblend_sources, segmentation
from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve

from pyBIA import data_processing, data_augmentation
from pyBIA.image_moments import make_moments_table
from pathlib import Path
from progress import bar

try:
    import scienceplots
    plt.style.use('science')
    plt.rcParams.update({'font.size':21,})
except:
    print('scienceplots not detected! Saved images will not be formatted correctly! Try: pip install scienceplots')

class Catalog:
    """
    Creates catalog object.
 
    Args:
        data (ndarray): 2D array.
        x (ndarray, optional): 1D array or list containing the x-pixel position.
            Can contain one position or multiple samples.
        y (ndarray, optional): 1D array or list containing the y-pixel position.
            Can contain one position or multiple samples.
        bkg (None, optional): If bkg=0 the data is assumed to be background-subtracted.
            The other optional is bkg=None, in which case the background will be
            automatically calculated for local regions.
        zp (float, optional):
        exptime (float, optional):
        error (ndarray, optional): 2D array containing the rms error map.
        morph_params (bool, optional): If True, image segmentation is performed and
            morphological parameters are computed. Defaults to True. 
        kernel_size (int): The size length of the square Gaussian filter kernel used to convolve 
            the data. This length must be odd. Defaults to 21.
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.7.
        threshold (float): Segmentation detection threshold. If during image segmentation no central source
            is located within a circular mask of radius = threshold, then the object is
            flagged as non-detection. Defaults to 10 pixels.
        deblend (bool, optional): If True, the objects are deblended during the segmentation
            procedure, thus deblending the objects before the morphological features
            are computed. Defaults to False so as to keep blobs as one segmentation object.
        obj_name (ndarray, str, optional): 1D array containing the name of each object
            corresponding to the x & y position. This will be appended to the first
            column of the output catalog. Defaults to None.
        field_name (ndarray, str, optional): 1D array containing the field name of each object
            corresponding to the x & y positions. This will be appended to the first
            column of the output catalog. Defaults to None.
        flag (ndarray, optional): 1D array containing a flag value for each object corresponding
            to the x & y positions. Defaults to None. 
        aperture (int): The radius of the photometric aperture. Defaults to 15.
        annulus_in (int): The inner radius of the circular aperture
            that will be used to calculate the background. Defaults to 20.
        annulus_out (int): The outer radius of the circular aperture
                that will be used to calculate the background. Defaults to 35.
        invert (bool, optional): If True, the x & y coordinates will be switched
            when cropping out the object during the image segmentation step. For
            more information see the morph_parameters function. Defaults to False.
        cat (Dataframe): Pandas dataframe, use if the catalog has been creted. The objects
            in this catalog must reside within the data array, therefore if subfields
            need individual class instance. Defaults to None.
    """

    def __init__(self, data, x=None, y=None, bkg=None, error=None, zp=None, exptime=None, morph_params=True, nsig=0.7, threshold=10, 
        deblend=False, obj_name=None, field_name=None, flag=None, aperture=15, annulus_in=20, annulus_out=35, 
        kernel_size=21, invert=False, cat=None):

        self.data = data 
        self.x = x
        self.y = y 
        self.bkg = bkg 
        self.error = error 
        self.zp = zp 
        self.exptime = exptime
        self.morph_params = morph_params
        self.kernel_size = kernel_size
        self.nsig = nsig
        self.threshold = threshold
        self.deblend = deblend
        self.aperture = aperture 
        self.annulus_in = annulus_in
        self.annulus_out = annulus_out
        self.kernel_size = kernel_size
        self.invert = invert 
        self.cat = cat

        #if bool(self.zp) != bool(self.exptime):
        #    raise ValueError('Both zp and exptime must be provided or not provided simultaneously!')

        if cat is not None:
            try:
                self.obj_name = np.array(cat['obj_name'])
            except:
                self.obj_name = obj_name
                pass

            try:
                self.field_name = np.array(cat['field_name'])
            except:
                self.field_name = field_name
                pass
                
            try:
                self.flag = np.array(cat['flag'])
            except:
                self.flag = flag
                pass

        elif self.cat is None:
            self.obj_name = obj_name
            self.field_name = field_name
            self.flag = flag

        if isinstance(x, np.ndarray) is False and x is not None:
            self.x = np.array(x)
        if isinstance(y, np.ndarray) is False and y is not None:
            self.y = np.array(y)
        if isinstance(obj_name, np.ndarray) is False and obj_name is not None:
            self.obj_name = np.array(obj_name)
        if isinstance(field_name, np.ndarray) is False and field_name is not None:
            self.field_name = np.array(field_name)
        if isinstance(flag, np.ndarray) is False and flag is not None:
            self.flag = np.array(flag)

    def create(self, save_file=True, path=None, filename=None):
        """
        Creates a photometric and morphological catalog containing the object(s) in 
        the given position(s) at the given order. The parameters x and y should be 1D 
        arrays containing the pixel location of each source. The input can be for a 
        single source or multiple objects.

        If data is background subtracted, set bkg=0. If no positions are input
        then a catalog is automatically generated using the segmentation threshold
        parameters (nsig). 

        Args:
            save_file (bool): If set to False then the catalog will not be saved to the machine. 
                You can always save manually, for example, if df = catalog(), then you can save 
                with: df.to_csv('filename'). Defaults to True.
            path (str, optional): By default the text file containing the photometry will be
                saved to the local directory, unless an absolute path to a directory is entered here.
            filename (str, optional): Name of the output catalog. Default name is 'pyBIA_catalog'.

        Note:
            As Lyman-alpha nebulae are diffuse sources with
            extended emission features, the default radius of
            the circular photometric aperture is 15 pixels. This 
            large aperture allows us to encapsulate the largest blobs.
        
            The background is calculated as the median pixel value
            within the area of the annulus. Increasing the size of the
            annulus may yield more robust background measurements. This
            is especially important when extracting photometry in crowded fields
            where surrounding sources may skew the median background.
                    
        Returns:
            A pandas dataframe of all objects input (or automatically detected if there were no position arguments), 
            containing both photometric and morphological information. A CSV file titled "pyBIA_catalog" 
            will also be saved to the local directory, unless an absolute path argument is specified.
        """

        if self.bkg is not None and self.bkg != 0:
            raise ValueError('Invalid background input -- if data is background subtracted set bkg=0, otherwise if bkg=None the background will be approximated.')
        if self.error is not None:
            if self.data.shape != self.error.shape:
                raise ValueError("The rms error map must be the same shape as the data array.")
        if self.aperture > self.annulus_in or self.annulus_in > self.annulus_out:
            raise ValueError('The radius of the inner and out annuli must be larger than the aperture radius.')
        if self.x is not None:
            try: #If position array is a single number it will be converted to a list of unit length
                __ = len(self.x)
            except:
                self.x, self.y = [self.x], [self.y]
            if len(self.x) != len(self.y):
                raise ValueError("The two position arrays (x & y) must be the same size.")
        #if self.invert == False:
        #    print('WARNING: If data is from .fits file you may need to set invert=True if (x,y) = (0,0) is at the top left corner of the image instead of the bottom left corner.')
        
        Ny, Nx = self.data.shape

        if self.x is None: #Background subtraction and source detection
            if self.nsig < 1 and self.deblend == False:
                warn('Low nsig warning, for proper source detection do an initial run with a higher nsig, or set deblend=True.')
            print('Running source detection...')
            length = self.annulus_out*2*2. #The sub-array when padding will be be a square encapsulating the outer annuli
            if Nx < length or Ny < length: #Small image, no need to pad, just take robust median
                if self.bkg is None:
                    data = self.data - sigma_clipped_stats(self.data)[1] #Sigma clipped median
                elif self.bkg == 0:
                    data = self.data
            else:
                if self.bkg is None:
                    data = subtract_background(self.data, length=length)
                elif self.bkg == 0:
                    data = self.data 
       
            segm, convolved_data = segm_find(data, nsig=self.nsig, kernel_size=self.kernel_size, deblend=self.deblend)
            props = segmentation.SourceCatalog(data, segm, convolved_data=convolved_data)
            try:
                self.x, self.y = props.centroid[:,0], props.centroid[:,1]
            except:
                self.x, self.y = props.centroid[0], props.centroid[1]
            print('{} sources detected!'.format(len(self.x)))

            positions = []
            for i in range(len(self.x)):
                positions.append((self.x[i], self.y[i]))

            aper_stats = ApertureStats(self.data, CircularAperture(positions, r=self.aperture), error=self.error)

            flux_err = None if self.error is None else aper_stats.sum_err

            if self.morph_params == True:
                prop_list, moment_list, self.segm_map = morph_parameters(data, self.x, self.y, exptime=self.exptime, nsig=self.nsig, kernel_size=self.kernel_size, median_bkg=None, 
                    invert=self.invert, deblend=self.deblend)
                tbl = make_table(prop_list, moment_list)
                self.cat = make_dataframe(table=tbl, x=self.x, y=self.y, zp=self.zp, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag,
                    flux=aper_stats.sum, flux_err=flux_err, median_bkg=None, save=save_file, path=path, filename=filename)
                
                return 

            self.cat = make_dataframe(table=None, x=self.x, y=self.y, zp=self.zp, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, 
                flux=aper_stats.sum, flux_err=flux_err, median_bkg=None, save=save_file, path=path, filename=filename)
            
            return 

        positions = []
        for i in range(len(self.x)):
            positions.append((self.x[i], self.y[i]))

        apertures = CircularAperture(positions, r=self.aperture)
        aper_stats = ApertureStats(self.data, apertures, error=self.error)

        if self.bkg is None:
            annulus_apertures = CircularAnnulus(positions, r_in=self.annulus_in, r_out=self.annulus_out)
            bkg_stats = ApertureStats(self.data, annulus_apertures, error=self.error, sigma_clip=SigmaClip())
            background = bkg_stats.median
            flux = aper_stats.sum - (background * apertures.area)
        elif self.bkg == 0:
            flux = aper_stats.sum 
            background = None 

        if self.error is None:
            if self.morph_params == True:
                prop_list, moment_list, self.segm_map = morph_parameters(self.data, self.x, self.y, exptime=self.exptime, nsig=self.nsig, kernel_size=self.kernel_size, median_bkg=background, 
                    invert=self.invert, deblend=self.deblend, threshold=self.threshold)
                tbl = make_table(prop_list, moment_list)
                self.cat = make_dataframe(table=tbl, x=self.x, y=self.y, zp=self.zp, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag,
                    flux=flux, median_bkg=background, save=save_file, path=path, filename=filename)
                return 

            self.cat = make_dataframe(table=None, x=self.x, y=self.y, zp=self.zp, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, 
                flux=flux, median_bkg=background, save=save_file, path=path, filename=filename)
            return 
           
        if self.morph_params == True:
            try:
                prop_list, moment_list, self.segm_map = morph_parameters(self.data, self.x, self.y, exptime=self.exptime, nsig=self.nsig, kernel_size=self.kernel_size, median_bkg=background, 
                        invert=self.invert, deblend=self.deblend, threshold=self.threshold)
            except AttributeError:
                raise ValueError(f'Could not compute morphological parameters with nsig={self.nsig}, as no segmentation patch could be generated. Reduce the value of nsig and try again.')
            
            tbl = make_table(prop_list, moment_list)
            self.cat = make_dataframe(table=tbl, x=self.x, y=self.y, zp=self.zp, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, 
                flux=flux, flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path, filename=filename)
            return 

        self.cat = make_dataframe(table=None, x=self.x, y=self.y, zp=self.zp, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, flux=flux, 
            flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path, filename=filename)
        return 

    def plot(self, index=None, obj_name=None, name='', pix_conversion=5, size=100):
        """
        Outputs two subplots, the image and the segmentation object.

        This class method assumes data is background subtracted at this point, otherwise
        uses the plot_segm() function which can subtract the background.

        Note:
            If obj_name and index are both None, then the whole data array will be plotted.

        Args:
            index (int, optional): Catalog index of the source to plot. Defaults to None.
            obj_name (str, optional): Source name, only works if the dataframe contains an 'obj_name' column, otherwise
                filter by index. Defaults to None.
            name (str, optional): Title of the plot. 

        Returns:
            AxesImage.
        """
            
        if index is None and obj_name is None:
            plot_segm(self.data, nsig=self.nsig, kernel_size=self.kernel_size, invert=self.invert, 
                median_bkg=self.bkg, deblend=self.deblend, name=name, pix_conversion=pix_conversion,
                r_in=self.annulus_in, r_out=self.annulus_out)
        else:
            if len(self.cat.shape) == 2:
                mask = np.where(self.cat['obj_name'] == obj_name)[0] if obj_name is not None else int(index)
                data = self.cat.iloc[mask]
                xpix, ypix = float(data['xpix']), float(data['ypix'])
            else:
                xpix, ypix = float(self.cat['xpix']), float(self.cat['ypix'])

            plot_segm(self.data, xpix=xpix, ypix=ypix, median_bkg=self.bkg, nsig=self.nsig, 
                kernel_size=self.kernel_size, invert=self.invert, deblend=self.deblend, name=name,
                r_in=self.annulus_in, r_out=self.annulus_out, size=100, pix_conversion=pix_conversion)
        return

def morph_parameters(data, x, y, size=100, nsig=0.6, threshold=10, kernel_size=21, median_bkg=None, 
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
       
        segm, convolved_data = segm_find(new_data, nsig=nsig, kernel_size=kernel_size, deblend=deblend)
        try:
            props = segmentation.SourceCatalog(new_data, segm, convolved_data=convolved_data)
        except:
            prop_list.append(-999), moment_list.append(-999) #If there are no segmented objects in the image
            progess_bar.next()
            continue

        # Mask a circular area at the center of the image, using radius=threshold
        # Flag if there is no segmented object within the circular mask 
        x_pos, y_pos = np.ogrid[:new_data.shape[0],:new_data.shape[1]]
        cx = cy = int(size/2)
        tmin, tmax = np.deg2rad((0,360))
        r2 = (x_pos-cx)*(x_pos-cx) + (y_pos-cy)*(y_pos-cy)
        theta = np.arctan2(x_pos-cx,y_pos-cy) - tmin
        theta %= (2*np.pi)
        circmask = r2 <= threshold*threshold
        anglemask = theta <= (tmax-tmin)
        mask = circmask*anglemask
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

    moment_list = ['m00','m10','m01','m20','m11','m02','m30','m21','m12','m03',             
        'mu10', 'mu01', 'mu20','mu11','mu02','mu30','mu21','mu12','mu03', 'hu1','hu2',
        'hu3','hu4','hu5','hu6','hu7', 'fourier_1','fourier_2','fourier_3',             
        'g10', 'g01', 'g20','g11','g02','g30','g21','g12','g03'] #Removes mu00

    prop_list = ['area', 'covar_sigx2', 'covar_sigy2', 'covar_sigxy', 'covariance_eigvals', 
        'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 
        'fwhm', 'gini', 'orientation', 'perimeter', 'semimajor_sigma', 'semiminor_sigma', #\\
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

    prop_list = ['m00','m10','m01','m20','m11','m02','m30','m21','m12','m03',             
        'mu10', 'mu01', 'mu20','mu11','mu02','mu30','mu21','mu12','mu03', 
        'hu1','hu2', 'hu3','hu4','hu5','hu6','hu7', 'fourier_1','fourier_2','fourier_3',             
        'g10', 'g01', 'g20','g11','g02','g30','g21','g12','g03', 'area', 'covar_sigx2', 
        'covar_sigy2', 'covar_sigxy', 'covariance_eigval1', 'covariance_eigval2', 
        'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 
        'fwhm', 'gini', 'orientation', 'perimeter', 'semimajor_sigma', 'semiminor_sigma', #\\
        'isscalar', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'max_value', 'maxval_xindex', 
        'maxval_yindex', 'min_value', 'minval_xindex', 'minval_yindex']

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

def DAO_find(data, fwhm):
    """
    Applyies DAOFIND algorithm (Stetson 1987) to detect sources in the image.

    Arg:
        data (ndarray): 2D array of a single image.
        fwhm (int): The full width at half maximum to use when . 
            Default is 14.

    Returns:
        First output is x-pixel array, second output is y-pixel array.
    """

    median, std = sigma_clipped_stats(data)[1:]
    print('Performing source detection...')
    daofind = DAOStarFinder(threshold=4.*std, fwhm=fwhm)  
    sources = daofind(data - median)
    try:
        index = np.where((sources['ycentroid']<0) | (sources['xcentroid']<0))[0]
    except:
        print('No objects found! Perhaps the fwhm is too low, default is fwhm=9.')
        return None
    sources = np.delete(sources, index)
    x, y = np.array(sources['xcentroid']), np.array(sources['ycentroid'])
    print('{} objects found!'.format(len(x)))
    
    return x, y

def segm_find(data, nsig=0.6, kernel_size=21, deblend=False, npixels=9, connectivity=8):
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
    sigma = 9.0 * gaussian_fwhm_to_sigma   # FWHM = 9. smooth the data with a 2D circular Gaussian kernel with a FWHM of 3 pixels to filter the image prior to thresholding:
    kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size, mode='center')
    convolved_data = convolve(data, kernel, normalize_kernel=True, preserve_nan=True)
    segm = detect_sources(convolved_data, threshold, npixels=npixels, connectivity=connectivity)
    if deblend is True:
        segm = deblend_sources(convolved_data, segm, npixels=npixels)
    
    return segm, convolved_data 

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

def plot_segm(data, xpix=None, ypix=None, size=100, median_bkg=None, nsig=0.7, kernel_size=21, invert=False,
    deblend=False, pix_conversion=5, r_in=20, r_out=35, cmap='viridis', path=None, name='', savefig=False, dpi=300, return_segm_only=False, npixels=9, connectivity=8):
    """
    Returns two subplots: source and corresponding segementation object. 

    If no x & y positions are input, the whole image will be used. If there are
    position areguments then a subimage of area size x size will be cropped out 
    first, centered around a given (x, y). By default size=100, although this should
    be a window size that comfortably encapsulates all objects. If this is too large
    the automatic background measurements will be less robust.

    Note:
        If savefig=True, the image will not outpout, it will only be saved. If path=None
        the .png will be saved to the local home directory.

        If data is backgrond subtracted, set median_bkg = 0.

    Example:
        If the data is background subtracted, to plot the entire image and the corresponding
        segmentation objects, we can do the following:

        >>> from pyBIA.catalog import plot_segm
        >>> plot_segm(data, median_bkg=0)

        If we wish to isolate a certain object, we input the (x,y) position(s) and a window size.
        If x & y are arrays that contain multiple positions, then each will plot in sequence.
        Since image data from .fits files has the (0,0) position on the top left, we will 
        also set invert=True so that the object is cropped out at the correct pixel position:

        >>> plot_segm(data, x, y, size=50, median_bkg=0, invert=True)

    Args:
        data (ndarray): 2D array of a single image.
        xpix (ndarray, optional): 1D array or list containing the x-pixel position.
            Can contain one position or multiple samples. Defaults to None, in which case
            the whole image is plotted.
        ypix (ndarray, optional): 1D array or list containing the y-pixel position.
            Can contain one position or multiple samples. Defaults to None, in which case
            the whole image is plotted.
        size (int): length/width of the output image. Defaults to
            100 pixels or data.shape[0] if image is small.
        median_bkg (ndarray, optional): If None then the median background will be
            subtracted using the median value within an annuli around the source. 
            If data is already background subtracted set median_bkg = 0. If this is
            an array, it must contain the local medium background around each source as
            this scalar will be subtracted locally.        
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.7.
        kernel_size (int): The size lenght of the square Gaussian filter kernel used to convolve 
            the data. This length must be odd. Defaults to 21.
        invert (bool): If True the x & y coordinates will be switched
            when cropping out the object, see Note below. Defaults to False. 
        deblend (bool, optional): If True, the objects are deblended during the segmentation
            procedure, thus deblending the objects before the morphological features
            are computed. Defaults to False so as to keep blobs as one segmentation object.
        pix_conversion (int): Pixels per arcseconds conversion factor. This is used to set
            the image axes. 
        r_in (int): Radius of first annuli used to compute the background, defaults to 20.
        r_out (int): Radius of second annuli used to compute the background, defaults to 35.
        cmap (str): Colormap to use when generating the image.
        path (str, optional): By default the text file containing the photometry will be
            saved to the local directory, unless an absolute path to a directory is entered here.
        name (str, ndarray, optional): The title of the image. This can be an array containing
            multiple names, in which case it must contain a name for each object.
        savefig (bool, optional): If True the plot will be saved to the specified
        dpi (int, optional): Dots per inch (resolution) when savefig=True. 
            Set dpi='figure' to use the image's dpi. Defaults to 300.
        return_segm_only (bool, optional): If True will not plot but instead return the segmentation map.
       
    Returns:
        AxesImage.

    """

    if data.max() == 0 and data.min() == 0:
        raise ValueError('The input data array contains only zeroes!')
    
    if median_bkg != 0 and median_bkg is not None:
        if isinstance(median_bkg, np.ndarray) is False:
            median_bkg = np.array(median_bkg)

    if len(data.shape) != 2:
        raise ValueError('Data must be 2 dimensional, 3d images not currently supported.')
    try:
        if len(median_bkg) != len(xpix):
            raise ValueError('If more than one median_bkg is input, the size of the array must be the number of sources (xpix, ypix) input.')
    except:
        pass

    if data.shape[1] < size:
        size = data.shape[1]

    #if data.shape[1] < 70: 
    #    r_out = data.shape[1]-1
    #    r_in = r_out - 15 #This is a 15 pixel annulus. 
    #elif data.shape[1] >= 72:
    #    r_out = 35
    #    r_in = 20 #Default annulus used to calculate the median bkg
    
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

        if return_segm_only:
            try:
                _ = segm.data
                return segm.data
            except AttributeError:
                print(f"WARNING: No segmentation patches could be generated anywhere on the image for sigma={nsig}, kernel size={kernel_size}, npixels={npixels}, and connectivity={connectivity}. Adjust the detection settings and try again! Returning zero-like array...")
                return np.zeros((size, size))


        props = segmentation.SourceCatalog(new_data, segm, convolved_data=convolved_data)
        #return segm.data
        with plt.rc_context({'axes.edgecolor':'silver', 'axes.linewidth':5, 'xtick.color':'black', 
            'ytick.color':'black', 'figure.facecolor':'white', 'axes.titlesize':22}):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1 = plt.subplot(2,1,1)
            ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
            #plt.tight_layout()
            index = np.where(np.isfinite(new_data))
            std = np.median(np.abs(new_data[index] - np.median(new_data[index])))
            vmin, vmax = np.median(new_data[index]) - 3*std, np.median(new_data[index]) + 10*std
            ax1.imshow(new_data, vmin=vmin, vmax=vmax, cmap=cmap)
            ax2.imshow(segm.data, origin='lower', cmap=segm.make_cmap(seed=19))
            #'seismic', 'twilight', 'YlGnBu_r', 'bone', 'cividis' #best cmaps
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams['figure.figsize'] = 5, 7
            plt.rcParams['xtick.major.pad'] = 6
            plt.rcParams['ytick.major.pad'] = 6

            plt.gcf().set_facecolor("black")
            plt.subplots_adjust(wspace=0, hspace=0)
            ax1.grid(True, color='k', alpha=0.35, linewidth=1.5, linestyle='--')
            ax2.grid(True, alpha=0.35, linestyle='--')

            ax1.tick_params(axis="both", which="both", colors="white", direction="in", labeltop=True,
                labelright=True, length=10, width=2, bottom=True, top=True, left=True, right=True, labelsize=12)
            ax2.tick_params(axis="both", which="both", colors="white", direction="in", labelbottom=True,
                labelleft=True, length=10, width=2, bottom=True, top=True, left=True, right=True, labelsize=12)

            for axis in ['right', 'left', 'bottom', 'top']:
                ax1.spines[axis].set_color("silver")
                ax1.spines[axis].set_linewidth(0.95)
                ax1.spines[axis].set_visible(True)
                ax2.spines[axis].set_color("silver")
                ax2.spines[axis].set_linewidth(0.95)
                ax2.spines[axis].set_visible(True)

            ax1.xaxis.set_visible(True)
            ax1.yaxis.set_visible(True)
            ax2.xaxis.set_visible(True)
            ax2.yaxis.set_visible(True)
            if isinstance(name, str):
                ax1.set_ylabel(name, color='mediumturquoise', loc='top', size=30)
            else:
                ax1.set_ylabel(name[i], color='mediumturquoise', loc='top', size=30)

            ax2.set_xlabel(r'$\Delta\alpha$ [arcsec]', fontweight='ultralight', color='snow', size=18)
            ax2.set_ylabel(r'$\Delta\delta$ [arcsec]', fontweight='ultralight', color='snow', size=18)
            
            ax1.yaxis.tick_right()
            ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax1.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax1.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax2.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax2.yaxis.set_minor_locator(tck.AutoMinorLocator(2))

            length = new_data.shape[0]
            x_label_list = [str(length/-2./pix_conversion), str(length/-4./pix_conversion), 0, str(length/4./pix_conversion), str(length/2./pix_conversion)]
            ticks = [0,length-3*length/4,length-length/2,length-length/4,length]
            x_label_list = [str(np.round(float(t), 1)) for t in x_label_list]
            
            ax1.set_frame_on(True)
            ax1.set_xticks(ticks)
            ax1.set_xticklabels(x_label_list)
            ax1.set_yticks(ticks)
            ax1.set_yticklabels(x_label_list)
            ax2.set_frame_on(True)
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(x_label_list)
            ax2.set_yticks(ticks)
            ax2.set_yticklabels(x_label_list)

            ax1.tick_params(axis="both", which='minor', length=5, color='w', direction='in')
            ax2.tick_params(axis="both", which='minor', length=5, color='w', direction='in')

            if savefig is True:
                if path is None:
                    print("No path specified, saving catalog to local home directory.")
                    path = str(Path.home())+'/'
                fig.savefig(path+name+'.png', dpi=dpi, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                

def plot_two_filters(data1, data2, xpix=None, ypix=None, size=100, median_bkg1=0, median_bkg2=0, nsig=[0.5, 0.5], kernel_size=21, invert=False,
    deblend=False, pix_conversion=5, r_in=20, r_out=35, cmap='viridis', path=None, filter1='Bw', filter2='Rw', name='Source Detection Threshold', savefig=False):
    """
    This is the function used to plot the same object in two different bands.

    If no x & y positions are input, the whole image will be used. If there are
    position areguments then a subimage of area size x size will be cropped out 
    first, centered around a given (x, y). By default size=100, although this should
    be a window size that comfortably encapsulates all objects. If this is too large
    the automatic background measurements will be less robust.

    Note:
        If median_bkg1 is None both images will be assumed to not be background subtracted!

    Args:
        data (ndarray): 2D array of a single image.
        xpix (ndarray, optional): 1D array or list containing the x-pixel position.
            Can contain one position or multiple samples. Defaults to None, in which case
            the whole image is plotted.
        ypix (ndarray, optional): 1D array or list containing the y-pixel position.
            Can contain one position or multiple samples. Defaults to None, in which case
            the whole image is plotted.
        size (int): length/width of the output image. Defaults to
            100 pixels or data.shape[0] if image is small.
        median_bkg (ndarray, optional): If None then the median background will be
            subtracted using the median value within an annuli around the source. 
            If data is already background subtracted set median_bkg = 0. If this is
            an array, it must contain the local medium background around each source as
            this scalar will be subtracted locally.        
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.7.
        kernel_size (int): The size lenght of the square Gaussian filter kernel used to convolve 
            the data. This length must be odd. Defaults to 21.
        invert (bool): If True the x & y coordinates will be switched
            when cropping out the object, see Note below. Defaults to False. 
        deblend (bool, optional): If True, the objects are deblended during the segmentation
            procedure, thus deblending the objects before the morphological features
            are computed. Defaults to False so as to keep blobs as one segmentation object.
        pix_conversion (int): Pixels per arcseconds conversion factor. This is used to set
            the image axes. 
        cmap (str): Colormap to use when generating the image.
        path (str, optional): By default the text file containing the photometry will be
            saved to the local directory, unless an absolute path to a directory is entered here.
        filter1 (str, ndarray, optional): The title of the image. This can be an array containing
            multiple names, in which case it must contain a name for each object.
        savefig (bool, optional): If True the plot will be saved to the specified
        dpi (int, optional): Dots per inch (resolution) when savefig=True. 
            Set dpi='figure' to use the image's dpi. Defaults to 300.
       
    Returns:
        AxesImage.

    """

    if data1.max() == 0 and data1.min() == 0:
        raise ValueError('The input data1 array contains only zeroes!')

    if data2.max() == 0 and data2.min() == 0:
        raise ValueError('The input data2 array contains only zeroes!')

    if isinstance(nsig, np.ndarray) is False:
        nsig = np.array(nsig)

    if median_bkg1 != 0 and median_bkg1 is not None:
        if isinstance(median_bkg1, np.ndarray) is False:
            median_bkg1 = np.array(median_bkg1)
    if median_bkg2 != 0 and median_bkg2 is not None:
        if isinstance(median_bkg2, np.ndarray) is False:
            median_bkg2 = np.array(median_bkg2)

    if len(data1.shape) != 2:
        raise ValueError('Data must be 2 dimensional, 3d images not currently supported.')
    try:
        if len(median_bkg1) != len(xpix):
            raise ValueError('If more than one median_bkg1 is input, the size of the array must be the number of sources (xpix, ypix) input.')
        if len(median_bkg2) != len(xpix):
            raise ValueError('If more than one median_bkg2 is input, the size of the array must be the number of sources (xpix, ypix) input.')
    except:
        pass

    if data1.shape[1] < size:
        size = data1.shape[1]

    #if data1.shape[1] < 70: 
    #    r_out = data1.shape[1]-1
    #    r_in = r_out - 15 #This is a 15 pixel annulus. 
    #elif data1.shape[1] >= 72:
    #    r_out = 35
    #    r_in = 20 #Default annulus used to calculate the median bkg
    
    if xpix is None and ypix is None:
        xpix, ypix = data1.shape[1]/2, data1.shape[1]/2
        size = data1.shape[1]

    try: 
        __ = len(xpix)
    except:
        xpix = [xpix]
    try:
        __ = len(ypix)
    except:
        ypix = [ypix]
    try:
        __ = len(median_bkg1)
    except:
        if median_bkg1 is not None:
            median_bkg1 = [median_bkg1]
    try:
        __ = len(median_bkg2)
    except:
        if median_bkg2 is not None:
            median_bkg2 = [median_bkg2]
   
    for i in range(len(xpix)):
        if size == data1.shape[1]:
            new_data1 = data1
            new_data2 = data2
        else: 
            new_data1 = data_processing.crop_image(data1, int(xpix[i]), int(ypix[i]), size, invert=invert)
            new_data2 = data_processing.crop_image(data2, int(xpix[i]), int(ypix[i]), size, invert=invert)

        if median_bkg1 is None: #Hard coding annuli size, inner:25 -> outer:35
            print("Subtracting background...")
            if new_data1.shape[0] > 200 and len(xpix) == 1:
                print('Calculating background in local regions, this will take a while... if data is background subtracted set median_bkg=0.')
                new_data1 = subtract_background(new_data1)
                new_data2 = subtract_background(new_data2)
            else:
                annulus_apertures = CircularAnnulus((new_data1.shape[1]/2, new_data1.shape[0]/2), r_in=r_in, r_out=r_out)
                bkg_stats = ApertureStats(new_data1, annulus_apertures, sigma_clip=SigmaClip())
                new_data1 -= bkg_stats.median
                bkg_stats = ApertureStats(new_data2, annulus_apertures, sigma_clip=SigmaClip())
                new_data2 -= bkg_stats.median
        elif median_bkg1 == 0:
            new_data1 -= median_bkg1
            new_data2 -= median_bkg2 
        else:
            new_data1 -= median_bkg1[i]
            new_data2 -= median_bkg2[i]

        segm1, convolved_data1 = segm_find(new_data1, nsig=nsig[0], kernel_size=kernel_size, deblend=deblend)
        segm2, convolved_data2 = segm_find(new_data2, nsig=nsig[1], kernel_size=kernel_size, deblend=deblend)

        return segm1.data, segm2.data

        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["font.family"] = "STIXGeneral"
        plt.rcParams["axes.formatter.use_mathtext"]=True
        plt.rcParams['text.usetex']=False

        #plt.style.use('/Users/daniel/Downloads/plot_style.txt')

        with plt.rc_context({'axes.edgecolor':'white', 'xtick.color':'white', 
            'ytick.color':'white', 'figure.facecolor':'black'}):#, 'axes.titlesize':22}):
            fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(8,8))
            fig.suptitle(name, x=.5, y=0.954, color='black', fontsize=22)
            ((ax1, ax2), (ax4, ax3)) = axes
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')
            ax4.set_aspect('equal')

            index = np.where(np.isfinite(new_data1))
            std = np.median(np.abs(new_data1[index] - np.median(new_data1[index])))
            vmin, vmax = np.median(new_data1[index]) - 3*std, np.median(new_data1[index]) + 10*std
            ax1.imshow(new_data1, vmin=vmin, vmax=vmax, cmap=cmap)

            index = np.where(np.isfinite(new_data2))
            std = np.median(np.abs(new_data2[index] - np.median(new_data2[index])))
            vmin, vmax = np.median(new_data2[index]) - 3*std, np.median(new_data2[index]) + 10*std
            ax2.imshow(new_data2, vmin=vmin, vmax=vmax, cmap=cmap)
            
            ax4.imshow(segm1.data, origin='lower', cmap=segm2.make_cmap(seed=19))
            ax3.imshow(segm2.data, origin='lower', cmap=segm2.make_cmap(seed=19))
            #
            #'seismic', 'twilight', 'YlGnBu_r', 'bone', 'cividis' #best cmaps
            
            plt.gcf().set_facecolor("white")
            fig.subplots_adjust(wspace=0, hspace=0)
            ax1.grid(True, color='k', alpha=0.35, linewidth=1.5, linestyle='--')
            ax2.grid(True, alpha=0.35, linestyle='--')
            ax3.grid(True, alpha=0.35, linestyle='--')
            ax4.grid(True, alpha=0.35, linestyle='--')

            for axis in ['right', 'left', 'bottom', 'top']:
                ax1.spines[axis].set_color("silver")
                ax1.spines[axis].set_linewidth(0.95)
                ax1.spines[axis].set_visible(True)
                ax2.spines[axis].set_color("silver")
                ax2.spines[axis].set_linewidth(0.95)
                ax2.spines[axis].set_visible(True)
                ax3.spines[axis].set_color("silver")
                ax3.spines[axis].set_linewidth(0.95)
                ax3.spines[axis].set_visible(True)
                ax4.spines[axis].set_color("silver")
                ax4.spines[axis].set_linewidth(0.95)
                ax4.spines[axis].set_visible(True)
            
            ax1.xaxis.set_visible(True)
            ax1.yaxis.set_visible(True)
            ax2.xaxis.set_visible(True)
            ax2.yaxis.set_visible(True)
            ax3.xaxis.set_visible(True)
            ax3.yaxis.set_visible(True)
            ax4.xaxis.set_visible(True)
            ax4.yaxis.set_visible(True)
            
            if isinstance(filter1, str):
                ax1.set_title(filter1, color='black', size=20)
            else:
                ax1.title(filter1[i], color='black', size=20)

            if isinstance(filter2, str):
                ax2.set_title(filter2, color='black', size=20)
            else:
                ax2.title(filter2[i], color='black', size=20)

            ax1.set_ylabel(r'$\Delta\delta \ \rm [arcsec]$', color='black', size=18)
            ax4.set_xlabel(r'$\Delta\alpha \ \rm [arcsec]$', color='black', size=18)
            ax4.set_ylabel(r'$\Delta\delta \ \rm [arcsec]$', color='black', size=18)
            ax3.set_xlabel(r'$\Delta\alpha \ \rm [arcsec]$', color='black', size=18)

            ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax4.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax4.xaxis.set_major_locator(plt.MaxNLocator(5))

            ax1.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax1.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax2.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax2.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax3.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax3.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax4.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
            ax4.yaxis.set_minor_locator(tck.AutoMinorLocator(2))

            length = new_data1.shape[0]
            x_label_list_1 = [str(length/-2./pix_conversion), str(length/-4./pix_conversion), 0, str(length/4./pix_conversion), str(length/2./pix_conversion)]
            ticks_1 = [0,length-3*length/4,length-length/2,length-length/4,length]
            x_label_list_1 = [str(np.round(float(t), 1)) for t in x_label_list_1]

            x_label_list_2 = [str(length/-4./pix_conversion), 0, str(length/4./pix_conversion)]
            ticks_2 = [length-3*length/4,length-length/2,length-length/4]
            x_label_list_2 = [str(np.round(float(t), 1)) for t in x_label_list_2]

            ax1.set_frame_on(True)
            ax1.set_xticks(ticks_2)
            ax1.set_xticklabels(x_label_list_2, color='black', fontsize=16)
            ax1.set_yticks(ticks_2)
            ax1.set_yticklabels(x_label_list_2, color='black', fontsize=16)

            ax2.set_frame_on(True)
            ax2.set_xticks(ticks_2)
            ax2.set_xticklabels(x_label_list_2, color='black', fontsize=16)
            ax2.set_yticks(ticks_2)
            ax2.set_yticklabels(x_label_list_2, color='black', fontsize=16)

            ax3.set_frame_on(True)
            ax3.set_xticks(ticks_2)
            ax3.set_xticklabels(x_label_list_2, color='black', fontsize=16)
            ax3.set_yticks(ticks_2)
            ax3.set_yticklabels(x_label_list_2, color='black', fontsize=16)

            ax4.set_frame_on(True)
            ax4.set_xticks(ticks_2)
            ax4.set_xticklabels(x_label_list_2, color='black', fontsize=16)
            ax4.set_yticks(ticks_2)
            ax4.set_yticklabels(x_label_list_2, color='black', fontsize=16)

            if savefig is True:
                if path is None:
                    print("No path specified, saving catalog to local home directory.")
                    path = str(Path.home())+'/'
                fig.savefig(path+name+'.png', dpi=300, bbox_inches='tight')
                plt.close()
                return
            plt.show()
            plt.close()




# FOR FIG1


def get_segmentation(data, nsig_val, pix_conversion, xpix=100, ypix=100, size=100, median_bkg=0, kernel_size=21, deblend=False, r_in=20, r_out=35, npixels=9, connectivity=8):
    """
    Calls the catalog module’s plot_segm to produce segmentation arrays.
    
    The function returns (segm1, segm2) where segm1 corresponds to data1.
    (In our usage, if only one image is available we pass the same image twice.)
    """

    segm1 = plot_segm(data, xpix=xpix, ypix=ypix, size=size, 
        median_bkg=median_bkg, nsig=nsig_val, kernel_size=kernel_size,
        deblend=deblend, pix_conversion=5, r_in=r_in, r_out=r_out, 
        return_segm_only=True, npixels=npixels, connectivity=connectivity)


    return segm1

def normalize_label(segm, center_val=50):
    """
    Normalize the segmentation array so that the connected component at the center
    (at index [center_val, center_val]) is set to 1000 and everything else to 0.
    """
    src_val = segm[center_val, center_val]
    if src_val == 0:
        return np.zeros(segm.shape) 

    segm[segm == src_val] = 1000
    segm[segm != 1000] = 0
    return segm

def compute_layered_segmentation(image, sigma_values, pix_conversion, xpix, ypix, size, median_bkg=0, kernel_size=21, deblend=False, r_in=20, r_out=35, npixels=9, connectivity=8):
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
        # For segmentation we call plot_segm.
        # If we have only one image, we pass it twice.
        segm = get_segmentation(data=image, nsig_val=sval, pix_conversion=pix_conversion, xpix=xpix, ypix=ypix, size=size, median_bkg=median_bkg, kernel_size=kernel_size, deblend=deblend, r_in=r_in, r_out=r_out, npixels=npixels, connectivity=connectivity)
        segm = normalize_label(segm)
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




#

#



#
def plot_objects_segmentation(
        *images,
        pix_conversion=1.0,
        sigma_values=[0.1, 0.25, 0.55, 0.95],
        titles=None,
        suptitle='',
        xpix=None,
        ypix=None,
        size=None,
        median_bkg=0, kernel_size=21, deblend=False, r_in=20, r_out=35, npixels=9, connectivity=8, cmap='viridis',
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
    # ------------------------------------------------------------------
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
            img = crop_image(img, xpix, ypix, size)

        extent = get_extent(img, pix_conversion)
        vmin, vmax = get_display_limits(img)
        layered = compute_layered_segmentation(
            img, sigma_values, pix_conversion,
            xpix if xpix is not None else 100,
            ypix if ypix is not None else 100,
            size if size is not None else 100,
            median_bkg=median_bkg, kernel_size=kernel_size, deblend=deblend, r_in=r_in, r_out=r_out, npixels=npixels, connectivity=connectivity
            )

        proc_imgs.append(img)
        extents.append(extent)
        vmins.append(vmin)
        vmaxs.append(vmax)
        layereds.append(layered)

    # ------------------------------------------------------------------
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

        # --- segmentation row ---
        ax_seg = fig.add_subplot(spec[1, idx])
        ax_seg.imshow(np.flip(layereds[idx], axis=0),
                      cmap='binary', vmin=0, vmax=1,
                      extent=extents[idx], origin='lower', interpolation='nearest')
        if idx == 0:
            ax_seg.set_ylabel(r'$\Delta \delta$ (arcsec)')
        else:
            ax_seg.set_yticklabels([])
        ax_seg.set_xlabel(r'$\Delta \alpha$ (arcsec)')

    # ------------------------------------------------------------------
    fig.suptitle(suptitle, y=1.09)

    default_intensities = [1.0, 0.7, 0.4, 0.1]
    intensities_used = default_intensities[:len(sigma_values)]
    legend_handles = [
        Patch(color=matplotlib.colors.to_hex(binary_cmap(inten)),
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






