# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 28 10:10:11 2021

@author: danielgodinez
"""
from astropy.utils.exceptions import AstropyWarning
from warnings import warn, filterwarnings, simplefilter
filterwarnings("ignore", category=AstropyWarning) #Ignore NaN & inf warnings
filterwarnings("ignore", category=RuntimeWarning) #Ignore sigma_clipping NaN warning
from astropy.io import fits 
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib.ticker as tck 
import numpy as np
import pandas as pd

from photutils.detection import DAOStarFinder
from photutils import detect_threshold, detect_sources, deblend_sources, segmentation
from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve

from pyBIA import data_processing
from pathlib import Path
from progress import bar

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
        error (ndarray, optional): 2D array containing the rms error map.
        morph_params (bool, optional): If True, image segmentation is performed and
            morphological parameters are computed. Defaults to True. 
        kernel_size (int): The size length of the square Gaussian filter kernel used to convolve 
            the data. This length must be odd. Defaults to 21.
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.7.
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

    def __init__(self, data, x=None, y=None, bkg=None, error=None, morph_params=True, nsig=0.7, deblend=False, 
        obj_name=None, field_name=None, flag=None, aperture=15, annulus_in=20, annulus_out=35, kernel_size=21,
        invert=False, cat=None):

        self.data = data 
        self.x = x
        self.y = y 
        self.bkg = bkg 
        self.error = error 
        self.morph_params = morph_params
        self.kernel_size = kernel_size
        self.nsig = nsig
        self.deblend = deblend
        self.aperture = aperture 
        self.annulus_in = annulus_in
        self.annulus_out = annulus_out
        self.kernel_size = kernel_size
        self.invert = invert 
        self.cat = cat
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
                len(self.x)
            except TypeError:
                self.x, self.y = [self.x], [self.y]
            if len(self.x) != len(self.y):
                raise ValueError("The two position arrays (x & y) must be the same size.")
        if self.invert == False:
            print('WARNING: If data is from .fits file you may need to set invert=True if (x,y) = (0,0) is at the top left corner of the image instead of the bottom left corner.')
        
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
            if self.error is None:
                flux_err = None 
            else:
                flux_err = aper_stats.sum_err

            if self.morph_params == True:
                prop_list = morph_parameters(data, self.x, self.y, nsig=self.nsig, kernel_size=self.kernel_size, median_bkg=None, 
                    invert=self.invert, deblend=self.deblend)
                tbl = make_table(prop_list)
                self.cat = make_dataframe(table=tbl, x=self.x, y=self.y, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag,
                    flux=aper_stats.sum, flux_err=flux_err, median_bkg=None, save=save_file, path=path, filename=filename)
                return 

            self.cat = make_dataframe(table=None, x=self.x, y=self.y, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, 
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
                prop_list = morph_parameters(self.data, self.x, self.y, nsig=self.nsig, kernel_size=self.kernel_size, median_bkg=background, 
                    invert=self.invert, deblend=self.deblend)
                tbl = make_table(prop_list)
                self.cat = make_dataframe(table=tbl, x=self.x, y=self.y, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag,
                    flux=flux, median_bkg=background, save=save_file, path=path, filename=filename)
                return 

            self.cat = make_dataframe(table=None, x=self.x, y=self.y, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, 
                flux=flux, median_bkg=background, save=save_file, path=path, filename=filename)
            return 
           
        if self.morph_params == True:
            prop_list = morph_parameters(self.data, self.x, self.y, nsig=self.nsig, kernel_size=self.kernel_size, median_bkg=background, 
                    invert=self.invert, deblend=self.deblend)
            tbl = make_table(prop_list)
            self.cat = make_dataframe(table=tbl, x=self.x, y=self.y, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, 
                flux=flux, flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path, filename=filename)
            return 

        self.cat = make_dataframe(table=None, x=self.x, y=self.y, obj_name=self.obj_name, field_name=self.field_name, flag=self.flag, flux=flux, 
            flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path, filename=filename)
        return 

    def plot(self, index=None, obj_name=None, name=''):
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
                median_bkg=self.bkg, deblend=self.deblend, name=name)
        else:
            if len(self.cat.shape) == 2:
                if obj_name is not None:
                    mask = np.where(self.cat['obj_name'] == obj_name)[0]
                else:
                    mask = int(index)
                    
                data = self.cat.iloc[mask]
                xpix, ypix = float(data['xpix']), float(data['ypix'])
            else:
                xpix, ypix = float(self.cat['xpix']), float(self.cat['ypix'])

            plot_segm(self.data, xpix=xpix, ypix=ypix, median_bkg=self.bkg, nsig=self.nsig, 
                kernel_size=self.kernel_size, invert=self.invert, deblend=self.deblend, name=name)
        return

def morph_parameters(data, x, y, size=100, nsig=0.6, kernel_size=21, median_bkg=None, invert=False, deblend=False):
    """
    Applies image segmentation on each object to calculate morphological 
    parameters calculated from the moment-based properties (see: ). These parameters 
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
            features related to the local environment. Default is 100 pixels, which
            is also the default image size the image classifier uses.
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.6.
        median_bkg (ndarray, optional): 1D array containing the median background
            in the annuli of each around each (x,y) object. This is not a standard rms or background
            map input. Defaults to None, in which case data is assumed to be background-subtracted.
        invert (bool): If True the x & y coordinates will be switched
            when cropping out the object, see Note below. Defaults to False.
        deblend (bool, optional): If True, the objects are deblended during the segmentation
            procedure, thus deblending the objects before the morphological features
            are computed. Defaults to False so as to keep blobs as one segmentation object.

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
        warn('Small image warning: results may be unstable if the object does not fit entirely within the frame.')
    try: #If position array is a single number it will be converted into a list of unit length
        len(x)
    except TypeError:
        x, y = [x], [y]

    size = 100 if data.shape[0] > 100 else data.shape[0]
    prop_list=[]
    progess_bar = bar.FillingSquaresBar('Applying image segmentation...', max=len(x))

    for i in range(len(x)):
        new_data = data_processing.crop_image(data, int(x[i]), int(y[i]), size, invert=invert)
        if median_bkg is not None:
            new_data -= median_bkg[i] 
       
        segm, convolved_data = segm_find(new_data, nsig=nsig, kernel_size=kernel_size, deblend=deblend)
        try:
            props = segmentation.SourceCatalog(new_data, segm, convolved_data=convolved_data)
        except:
            prop_list.append(-999) #If the object can't be segmented 
            continue

        sep_list=[]
        for xx in range(len(props)): #This is to select the segmented object closest to the center, (x,y)=(50,50)
            xcen = float(props[xx].centroid[0])
            ycen = float(props[xx].centroid[1])
            sep_list.append(np.sqrt((xcen-(size/2))**2 + (ycen-(size/2))**2))

        inx = np.where(sep_list == np.min(sep_list))[0]
        if len(inx) > 1: #In case objects can't be deblended
            inx = inx[0] 

        prop_list.append(props[inx])
        progess_bar.next()
    progess_bar.finish()

    if -999 in prop_list:
        print('WARNING: At least one object could not be detected in segmentation... perhaps the object is too faint. This object is still in the catalog, the morphological features have been set to -999.')
    return np.array(prop_list, dtype=object)

def make_table(props):
    """
    Returns the morphological parameters calculated from the sementation image.
    A list of the parameters and their function is available in the Photutils
    Source Catalog documentation: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html
    
    Args:
        Props (source catalog): A source catalog containing the segmentation parameters.
        
    Returns:
        Array containing the morphological features. 

    """

    prop_list = ['area', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'bbox',
        'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'covariance_eigvals', 'cxx', 'cxy', 
        'cyy', 'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 'fwhm',
        'gini', 'isscalar', 'max_value', 'maxval_xindex', 'maxval_yindex', 'min_value', 
        'minval_xindex', 'minval_yindex', 'moments', 'moments_central', 'orientation', 
        'perimeter', 'segment_flux', 'semimajor_sigma', 'semiminor_sigma']

    table = []
    print('Writing catalog...')
    for i in range(len(props)):
        morph_feats = []
        try:
            props[i][0].area
        except:
            for j in range(len(prop_list)+31): #+31 because covariance eigenvalue param is actually 2 params, and the moments to 3rd order are each 16
                morph_feats.append(-999)
            table.append(morph_feats)
            continue

        QTable = props[i][0].to_table(columns=prop_list)
        for param in prop_list:
            if param == 'moments' or param == 'moments_central': #To 3rd order there are 16 image moments (4x4)
                for moment in np.ravel(QTable[param]):
                    morph_feats.append(moment)
            elif param == 'covariance_eigvals': #Data is 2D therefore we get two eigenvalues
                morph_feats.append(np.ravel(QTable[param])[0].value)
                morph_feats.append(np.ravel(QTable[param])[1].value)
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

def make_dataframe(table=None, x=None, y=None, flux=None, flux_err=None, median_bkg=None, 
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

        >>> props = morph_parameters(data, x=xpix, y=ypix)
        >>> table = make_table(props)
        >>> dataframe = make_dataframe(table, x=xpix, y=ypix)

    Returns:
        Pandas dataframe containing the parameters and features of all objects
        in the input data table. If save=True, a CSV file titled 'pybia_catalog'
        will be saved to the local directory, unless a path is specified.

    """

    if filename is None:
        filename = 'pyBIA_catalog'

    prop_list = ['area', 'bbox_xmax', 'bbox_xmin', 'bbox_ymax', 'bbox_ymin', 'bbox',
        'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'covariance_eigvals1', 'covariance_eigvals2',
        'cxx', 'cxy', 'cyy', 'eccentricity', 'ellipticity', 'elongation', 'equivalent_radius', 'fwhm',
        'gini', 'isscalar', 'max_value', 'maxval_xindex', 'maxval_yindex', 'min_value', 'minval_xindex', 
        'minval_yindex', 'moments_1', 'moments_2', 'moments_3', 'moments_4', 'moments_5', 'moments_6', 
        'moments_7', 'moments_8', 'moments_9', 'moments_10', 'moments_11', 'moments_12', 'moments_13', 
        'moments_14', 'moments_15', 'moments_16', 'moments_central_1', 'moments_central_2', 'moments_central_3', 
        'moments_central_4', 'moments_central_5', 'moments_central_6', 'moments_central_7', 'moments_central_8', 
        'moments_central_9', 'moments_central_10', 'moments_central_11', 'moments_central_12', 'moments_central_13', 
        'moments_central_14', 'moments_central_15', 'moments_central_16', 'orientation', 'perimeter', 'segment_flux', 
        'semimajor_sigma', 'semiminor_sigma']

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
        data_dict['flux'] = flux
    if flux_err is not None:
        data_dict['flux_err'] = flux_err
    
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
        len(table)
    except TypeError:
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

def segm_find(data, nsig=0.6, kernel_size=21, deblend=False):
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

    Returns:
        First output is the segmentation image object, the second output is the convolved data
        that was used when cataloging the segmentation objects.

    """

    threshold = detect_threshold(data, nsigma=nsig, background=0.0)
    sigma = 9.0 * gaussian_fwhm_to_sigma   # FWHM = 9. smooth the data with a 2D circular Gaussian kernel with a FWHM of 3 pixels to filter the image prior to thresholding:
    kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size, mode='center')
    convolved_data = convolve(data, kernel, normalize_kernel=True, preserve_nan=True)
    segm = detect_sources(convolved_data, threshold, npixels=9, kernel=None, connectivity=8)
    if deblend is True:
        segm = deblend_sources(convolved_data, segm, npixels=5, kernel=None)
    
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

def plot_segm(data, xpix=None, ypix=None, size=100, median_bkg=None, nsig=0.7, kernel_size=21, invert=False,
    deblend=False, pix_conversion=5, cmap='viridis', path=None, name='', savefig=False, dpi=300):
    """
    Returns two subplots: source and segementation object. 

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
        cmap (str): Colormap to use when generating the image.
        path (str, optional): By default the text file containing the photometry will be
            saved to the local directory, unless an absolute path to a directory is entered here.
        name (str, ndarray, optional): The title of the image. This can be an array containing
            multiple names, in which case it must contain a name for each object.
        savefig (bool, optional): If True the plot will be saved to the specified
        dpi (int, optional): Dots per inch (resolution) when savefig=True. 
            Set dpi='figure' to use the image's dpi. Defaults to 300.
       
    Returns:
        AxesImage.

    """
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
    if data.shape[1] < 70: 
        r_out = data.shape[1]-1
        r_in = r_out - 15 #This is a 15 pixel annulus. 
    elif data.shape[1] >= 72:
        r_out = 35
        r_in = 20 #Default annulus used to calculate the median bkg
    
    if xpix is None and ypix is None:
        xpix, ypix = data.shape[1]/2, data.shape[1]/2
        size = data.shape[1]

    try: 
        len(xpix)
    except TypeError:
        xpix = [xpix]
    try:
        len(ypix)
    except TypeError:
        ypix = [ypix]
    try:
        len(median_bkg)
    except TypeError:
        if median_bkg is not None:
            median_bkg = [median_bkg]
            
    for i in range(len(xpix)):
        if size == data.shape[1]:
            new_data = data
        else: 
            new_data = data_processing.crop_image(data, int(xpix[i]), int(ypix[i]), size, invert=invert)

        if median_bkg is None: #Hard coding annuli size, inner:25 -> outer:35
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

        segm, convolved_data = segm_find(new_data, nsig=nsig, kernel_size=kernel_size, deblend=deblend)
        props = segmentation.SourceCatalog(new_data, segm, convolved_data=convolved_data)

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

            ax2.set_xlabel(r'$\Delta\alpha$ [arcsec] ', fontweight='ultralight', color='snow', size=18)
            ax2.set_ylabel(r'$\Delta\delta$ [arcsec] ', fontweight='ultralight', color='snow', size=18)
            
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
                fig.savefig(path+name, dpi=dpi)
                continue
            plt.show()

### Standalone function for creating the catalog ###
def create_cat(data, x=None, y=None, bkg=None, error=None, morph_params=True, nsig=0.6, deblend=False, 
    obj_name=None, field_name=None, flag=None, aperture=15, annulus_in=20, annulus_out=35, 
    invert=False, save_file=True, path=None, filename=None):
    """
    Creates a photometric and morphological catalog containing the object(s) in 
    the given position(s) at the given order. The parameters x and y should be 1D 
    arrays containing the pixel location of each source. The input can be for a 
    single source or multiple objects.

    If data is background subtracted, set bkg=0. If no positions are input
    then a catalog is automatically generated using the segmentation threshold
    parameters (nsig). 

    Example:
        We can use the world coordinate system in astropy
        to convert ra/dec to pixels and then call pyBIA.catalog:

        >>> import astropy
        >>> from pyBIA import catalog

        >>> hdu = astropy.io.fits.open(name)
        >>> wcsobj= astropy.wcs.WCS(header = hdu[0].header)

        >>> x_pix, y_pix = wcsobj.all_world2pix(ra, dec, 0) 
        >>> catalog.create(data, x_pix, y_pix)

    Args:
        data (ndarray): 2D array.
        x (ndarray, optional): 1D array or list containing the x-pixel position.
            Can contain one position or multiple samples.
        y (ndarray, optional): 1D array or list containing the y-pixel position.
            Can contain one position or multiple samples.
        bkg (None, optional): If bkg=0 the data is assumed to be background-subtracted.
            The other optional is bkg=None, in which case the background will be
            automatically calculated for local regions.
        error (ndarray, optional): 2D array containing the rms error map.
        morph_params (bool, optional): If True, image segmentation is performed and
            morphological parameters are computed. Defaults to True. 
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.6.
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
        save_file (bool): If set to False then the catalog will not be saved to the machine. 
            You can always save manually, for example, if df = catalog(), then you can save 
            with: df.to_csv('filename'). Defaults to True.
        path (str, optional): By default the text file containing the photometry will be
            saved to the local directory, unless an absolute path to a directory is entered here.
        filename(str, optional): Name of the output catalog. Default name is 'pyBIA_catalog'.

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
        A catalog of all objects input (or automatically detected if there were no position arguments), 
        containing both photometric and morphological information. A CSV file titled "pyBIA_catalog" 
        will also be saved to the local directory, unless an absolute path argument is specified.
    """
    if isinstance(x, np.ndarray) is False:
        x = np.array(x)
    if isinstance(y, np.ndarray) is False:
        y = np.array(y)

    if bkg is not None and bkg != 0:
        raise ValueError('Invalid background input -- if data is background subtracted set bkg=0, otherwise if bkg=None the background will be approximated.')
    if error is not None:
        if data.shape != error.shape:
            raise ValueError("The rms error map must be the same shape as the data array.")
    if aperture > annulus_in or annulus_in > annulus_out:
        raise ValueError('The radius of the inner and out annuli must be larger than the aperture radius.')
    if x is not None:
        try: #If position array is a single number it will be converted to a list of unit length
            len(x)
        except TypeError:
            x, y = [x], [y]
        if len(x) != len(y):
            raise ValueError("The two position arrays (x & y) must be the same size.")
    if invert == False:
        warn('If data is from .fits file you may need to set invert=True if (x,y) = (0,0) is at the top left corner of the image instead of the bottom left corner.')
    """
    #Apply DAOFIND (Stetson 1987) to detect sources in the image
    if x is None: 
        x, y = DAO(data, fwhm)
        #print('{} objects found!'.format(len(x)))
    """
    if x is None: #Background approximation to create preliminary source positions
        Nx, Ny = data.shape[1], data.shape[0]
        length = annulus_out*2*2. #The sub-array when padding will be be a square encapsulating the outer annuli
        
        if Nx < length or Ny < length: #Small image, no need to pad just take robust median
            if bkg is None:
                data -= sigma_clipped_stats(data)[1] #Sigma clipped median
        else:
            if bkg is None:
                data = subtract_background(data, length=length)

        segm, convolved_data = segm_find(data, nsig=nsig, kernel_size=kernel_size, deblend=deblend)
        props = segmentation.SourceCatalog(data, segm, convolved_data=convolved_data)
        x, y = props.centroid[0], props.centroid[1]
     
    positions = []
    for j in range(len(x)):
        positions.append((x[j], y[j]))

    apertures = CircularAperture(positions, r=aperture)
    aper_stats = ApertureStats(data, apertures, error=error)

    if bkg is None:
        annulus_apertures = CircularAnnulus(positions, r_in=annulus_in, r_out=annulus_out)
        bkg_stats = ApertureStats(data, annulus_apertures, error=error, sigma_clip=SigmaClip())
        background = bkg_stats.median
        flux = aper_stats.sum - (background * apertures.area)
    elif bkg == 0:
        flux = aper_stats.sum 
        background = None 

    if error is None:
        if morph_params == True:
            prop_list = morph_parameters(data, x, y, nsig=nsig, median_bkg=background, invert=invert, deblend=deblend)
            tbl = make_table(prop_list)
            df = make_dataframe(table=tbl, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag,
                flux=flux, median_bkg=background, save=save_file, path=path, filename=filename)
            return df

        df = make_dataframe(table=None, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag, 
            flux=flux, median_bkg=background, save=save_file, path=path, filename=filename)
        return df
       
    if morph_params == True:
        prop_list = morph_parameters(data, x, y, nsig=nsig, median_bkg=background, invert=invert, deblend=deblend)
        tbl = make_table(prop_list)
        df = make_dataframe(table=tbl, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag, 
            flux=flux, flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path, filename=filename)
        return df

    df = make_dataframe(table=None, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag, flux=flux, 
        flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path, filename=filename)
    return df, prop_list
