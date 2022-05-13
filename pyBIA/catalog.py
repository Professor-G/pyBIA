# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 28 10:10:11 2021

@author: danielgodinez
"""
from astropy.utils.exceptions import AstropyWarning
from warnings import warn, filterwarnings, simplefilter
filterwarnings("ignore", category=AstropyWarning) #Ignore NaN & inf warnings
from astropy.io import fits 
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import matplotlib.ticker as tck 
import numpy as np
import pandas as pd

from photutils.detection import DAOStarFinder
from photutils import detect_threshold, detect_sources, segmentation
from photutils.aperture import ApertureStats, CircularAperture, CircularAnnulus
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve

from pyBIA import data_processing


def create(data, error=None, morph_params=True, deblend=False, x=None, y=None, obj_name=None, field_name=None, 
    flag=None, aperture=15, annulus_in=20, annulus_out=35, invert=False, fwhm=14, save_file=True, path=None, 
    filename=None):
    """
    Creates a photometric and morphological catalog containing the object(s) in 
    the given position(s) at the given order. The parameters x and y should be 1D 
    arrays containing the pixel location of each source. The input can be for a 
    single object or multiple objects.

    If no positions are input then a catalog is automatically 
    generated using the DAOFIND algorithm. Sources with local density
    maxima greater than 3 standard deviations from the background. 
    
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
        error (ndarray, optional): 2D array containing the rms error map.
        morph_params (bool, optional): If True, image segmentation is performed and
            morphological parameters are computed. Defaults to True. 
        deblend (bool, optional): If True, the objects are deblended during the segmentation
            procedure, thus deblending the objects before the morphological features
            are computed. Defaults to False so as to keep blobs as one segmentation object.
        x (ndarray, optional): 1D array or list containing the x-pixel position.
            Can contain one position or multiple samples.
        y (ndarray, optional): 1D array or list containing the y-pixel position.
            Can contain one position or multiple samples.
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
        fwhm (int): The circularized full width at half maximum (FWHM). 
            Default is 2.
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
    if x is None: #Apply DAOFIND (Stetson 1987) to detect sources in the image
        median, std = sigma_clipped_stats(data)[1:]
        print('Performing source detection...')
        daofind = DAOStarFinder(threshold=4.*std, fwhm=fwhm)  #Found this threshold and fwhm as ideal through trial and error using blobs for validation
        sources = daofind(data - median)
        try:
            index = np.where((sources['ycentroid']<0) | (sources['xcentroid']<0))[0]
        except:
            print('No objects found! Perhaps the fwhm is too low, default is fwhm=9.')
            return None
        sources = np.delete(sources, index)
        x, y = np.array(sources['xcentroid']), np.array(sources['ycentroid'])
        #print('{} objects found!'.format(len(x)))

    positions = []
    for j in range(len(x)):
        positions.append((x[j], y[j]))

    apertures = CircularAperture(positions, r=aperture)
    annulus_apertures = CircularAnnulus(positions, r_in=annulus_in, r_out=annulus_out)
    aper_stats = ApertureStats(data, apertures, error=error)
    bkg_stats = ApertureStats(data, annulus_apertures, error=error, sigma_clip=SigmaClip())
    
    background = bkg_stats.median  
    flux = aper_stats.sum - (background * apertures.area)

    if error is None:
        if morph_params == True:
            prop_list = morph_parameters(data, x, y, median_bkg=background, invert=invert, deblend=deblend)
            tbl = make_table(prop_list)
            df = make_dataframe(table=tbl, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag,
                flux=flux, median_bkg=background, save=save_file, path=path)
            return df

        df = make_dataframe(table=None, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag, 
            flux=flux, median_bkg=background, save=save_file, path=path)
        return df
       
    if morph_params == True:
        prop_list = morph_parameters(data, x, y, median_bkg=background, invert=invert, deblend=deblend)
        tbl = make_table(prop_list)
        df = make_dataframe(table=tbl, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag, 
            flux=flux, flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path)
        return df

    df = make_dataframe(table=None, x=x, y=y, obj_name=obj_name, field_name=field_name, flag=flag, flux=flux, 
        flux_err=aper_stats.sum_err, median_bkg=background, save=save_file, path=path)
    return df, prop_list

def morph_parameters(data, x, y, size=100, median_bkg=None, invert=False, deblend=False):
    """
    Applies image segmentation on each object to calculate morphological 
    parameters calculated from the moment-based properties. These parameters 
    can be used to train a machine learning classifier.

    By default the data is assumed to be background subtracted.
    
    Args:
        data (ndarray): 2D array.
        x (ndarray): 1D array or list containing the x-pixel position.
            Can contain one position or multiple samples.
        y (ndarray): 1D array or list containing the y-pixel position.
            Can contain one position or multiple samples.
        size (int, optional): The size of the box to consider when calculating
            features related to the local environment. Default is 100 pixels, which
            is also the default image size the image classifier uses.
        median_bkg (ndarray, optional): 1D array containing the median background
            around the annuli of each (x,y) object. This is not a standard rms or background
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

    print('Applying image segmentation, this could take a while...')
    size = 100 if data.shape[0] > 100 else data.shape[0]
    prop_list=[]

    for i in range(len(x)):
        new_data = data_processing.crop_image(data, int(x[i]), int(y[i]), size, invert=invert)
        if median_bkg is not None:
            new_data -= median_bkg[i] 

        threshold = detect_threshold(new_data, nsigma=0.6, background=0.0)
        kernel = Gaussian2DKernel(x_stddev=9*gaussian_fwhm_to_sigma, x_size=21, y_size=21)
        kernel.normalize()

        convolved_data = convolve(new_data, kernel, normalize_kernel=False, preserve_nan=True)
        segm = detect_sources(convolved_data, threshold, npixels=9, kernel=None, connectivity=8)
        if deblend is True:
            segm = segmentation.deblend_sources(convolved_data, segm, kernel=None, npixels=9, connectivity=8)

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

    if -999 in prop_list:
        warn('At least one object could not be detected in segmentation... perhaps the object is too faint or there is a coordinate error. This object is still in the catalog, the morphological features have been set to -999.')
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
        'gini', 'isscalar', 'kron_flux', 'kron_radius', 'max_value', 'maxval_xindex', 
        'maxval_yindex', 'min_value', 'minval_xindex', 'minval_yindex', 'moments', 
        'moments_central', 'orientation', 'perimeter', 'segment_flux', 'semimajor_sigma', 
        'semiminor_sigma']

    table = []
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
            elif param == 'covariance_eigvals': #In two dimensions there are 2 eigenvalues
                morph_feats.append(np.ravel(QTable[param])[0].value)
                morph_feats.append(np.ravel(QTable[param])[1].value)
            elif param == 'isscalar':
                if QTable[param] == True: #Checks whether it's a single source, 1 for true, 0 for false
                    morph_feats.append(1)
                else:
                    morph_feats.append(0)
            elif param == 'bbox': #Calculate area of rectangular bounding box
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
        'gini', 'isscalar', 'kron_flux', 'kron_radius', 'max_value', 'maxval_xindex', 
        'maxval_yindex', 'min_value', 'minval_xindex', 'minval_yindex', 'moments_1', 'moments_2',
        'moments_3', 'moments_4', 'moments_5', 'moments_6', 'moments_7', 'moments_8', 'moments_9', 'moments_10',
        'moments_11', 'moments_12', 'moments_13', 'moments_14', 'moments_15', 'moments_16', 'moments_central_1',
        'moments_central_2', 'moments_central_3', 'moments_central_4', 'moments_central_5', 'moments_central_6', 
        'moments_central_7', 'moments_central_8', 'moments_central_9', 'moments_central_10', 'moments_central_11', 
        'moments_central_12', 'moments_central_13', 'moments_central_14', 'moments_central_15', 'moments_central_16',
        'orientation', 'perimeter', 'segment_flux', 'semimajor_sigma', 'semiminor_sigma']

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
    if flux is not None:
        data_dict['flux'] = flux
    if flux_err is not None:
        data_dict['flux_err'] = flux_err
    if median_bkg is not None:
        data_dict['median_bkg'] = median_bkg
    
    if table is None:
        df = pd.DataFrame(data_dict)
        if save == True:
            if path is None:
                print("No path specified, saving catalog to local home directory.")
                path = '~/'
            df.to_csv(path+filename) 
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
            path = '~/'
        df.to_csv(path+filename) 
        return df
    return df

def plot_segm(data, xpix=None, ypix=None, size=100, median_bkg=None, nsig=0.6, kernel_size=21, invert=False,
    deblend=False, pix_conversion=5, cmap='viridis', path=None, name=' ', savefig=False):
    """
    Plots two subplots: source and segementation object. 

    If no x & y positions are input, the whole image will be output.

    Note:
        If savefig=True, the image will not plot, it will only be saved. If path=None
        the .png will be saved to the local home directory.

        If data is backgrond subtracted, set median_bkg = 0.

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
            If data is already background subtracted set median_bkg = 0.
        nsig (float): The sigma detection limit. Objects brighter than nsig standard 
            deviations from the background will be detected during segmentation. Defaults to 0.6.
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
        name (str, optional): Title displayed above the image. 
        savefig (bool, optional): If True the plot will be saved to the specified
       
    Returns:
        AxesImage.

    """
    if data.shape[0] > 500 and median_bkg is None and xpix is None:
        warn('Background subtraction is not as robust when the image is too large. This will also affect the data range the colormap covers.')
    if len(data.shape) != 2:
        raise ValueError('Data must be 2 dimensional, 3d images not currently supported.')

    if data.shape[0] < size:
       size = data.shape[0]
    if data.shape[0] < 36:
        r_out = data.shape[0]-1
        r_in = r_out - 10 #This is a 10 pixel annulus. 
    elif data.shape[0] >= 36:
        r_out = 35
        r_in = 25 #Default annulus used to calculate the median bkg
    
    if xpix is None and ypix is None:
        xpix, ypix = data.shape[1]/2, data.shape[0]/2
        size = data.shape[0]

    try: 
        len(xpix)
    except TypeError:
        xpix = [xpix]
    try:
        len(ypix)
    except TypeError:
        ypix = [ypix]

    for i in range(len(xpix)):
        if size != data.shape[0]: 
            new_data = data_processing.crop_image(data, int(xpix[i]), int(ypix[i]), size, invert=invert)
        else:
            new_data = data


        if median_bkg is None:
            annulus_apertures = CircularAnnulus((xpix[i], ypix[i]), r_in=r_in, r_out=r_out)
            bkg_stats = ApertureStats(data, annulus_apertures, sigma_clip=SigmaClip())
            median_bkg = bkg_stats.median  

        new_data -= median_bkg 

        threshold = detect_threshold(new_data, nsigma=nsig, background=0.0)
        sigma = 9.0 * gaussian_fwhm_to_sigma   # FWHM = 9. smooth the data with a 2D circular Gaussian kernel with a FWHM of 3 pixels to filter the image prior to thresholding:
        kernel = Gaussian2DKernel(sigma, x_size=kernel_size, y_size=kernel_size, mode='center')
        convolved_data = convolve(new_data, kernel, normalize_kernel=True, preserve_nan=True)
        segm = detect_sources(convolved_data, threshold, npixels=9, kernel=None, connectivity=8)
        if deblend is True:
            segm = deblend_sources(convolved_data, segm, npixels=5, kernel=None)
        props = segmentation.SourceCatalog(new_data, segm, convolved_data=convolved_data)

        with plt.rc_context({'axes.edgecolor':'silver', 'axes.linewidth':5, 'xtick.color':'black', 
            'ytick.color':'black', 'figure.facecolor':'white', 'axes.titlesize':22}):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)#, subplot_kw=dict(frameon=False))
            ax1 = plt.subplot(2,1,1)
            ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
            #plt.tight_layout()
            std = np.median(np.abs(data-np.median(data)))
            vmin, vmax = np.median(data) - 3*std, np.median(data) + 10*std
            ax1.imshow(np.flip(np.flip(data)), vmin=vmin, vmax=vmax, cmap=cmap)
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
            ax2.tick_params(axis="both", which="both", colors="r", direction="in", labelbottom=True,
                labelleft=True, length=10, width=2, bottom=True, top=True, left=True, right=True, labelsize=12)

            for axis in ['right', 'left', 'bottom', 'top']:
                ax1.spines[axis].set_color("silver")
                ax1.spines[axis].set_linewidth(0.95)
                ax1.spines[axis].set_visible(True)
                ax2.spines[axis].set_color("silver")
                ax2.spines[axis].set_linewidth(0.95)
                ax2.spines[axis].set_visible(True)

            ax1.xaxis.set_visible(True)
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

            ax1.tick_params(axis="both", which='minor', length=5, color='w', direction='in')
            ax2.tick_params(axis="both", which='minor', length=5, color='r', direction='in')

            size = new_data.shape[0]
            x_label_list = [str(size/-2./pix_conversion), str(size/-4./pix_conversion), 0, str(size/4./pix_conversion), str(size/2./pix_conversion)]
            ticks = [0,size-3*size/4,size-size/2,size-size/4,size]
            
            ax1.set_xticks(ticks)
            ax1.set_xticklabels(x_label_list)
            ax1.set_yticks(ticks)
            ax1.set_yticklabels(x_label_list)
            ax1.set_frame_on(True)
            ax2.set_frame_on(True)

            ax2.set_xticks(ticks)
            ax2.set_xticklabels(x_label_list)
            ax2.set_yticks(ticks)
            ax2.set_yticklabels(x_label_list)
            if savefig is True:
                if path is None:
                    print("No path specified, saving catalog to local home directory.")
                    path = '~/'
                fig.savefig(path+name, dpi=300)
                continue
            plt.show()    

ra = np.loadtxt('/Users/daniel/Desktop/NDWFS_Tiles/photometric_catalog/85/ndwfs_catalog_bw_85', usecols=0)
dec = np.loadtxt('/Users/daniel/Desktop/NDWFS_Tiles/photometric_catalog/85/ndwfs_catalog_bw_85', usecols=1)
field_name = np.loadtxt('/Users/daniel/Desktop/NDWFS_Tiles/photometric_catalog/85/ndwfs_catalog_bw_85', usecols=2, dtype=str)
indices = np.loadtxt('/Users/daniel/Desktop/NDWFS_Tiles/photometric_catalog/85/indices_bw_85')

indices = np.argsort(indices)
ra, dec, field_name = ra[indices], dec[indices], field_name[indices] #To match Moire's paper
i=7#5, 21 50 69
for i in range(15,16):
    fieldname = field_name[i]
    hdu = fits.open('/Users/daniel/Desktop/NDWFS_Tiles/Bw_FITS/'+field_name[i]+'_Bw_03_fix.fits')
    data = hdu[0].data 

    wcs = WCS(header = hdu[0].header)
    xpix, ypix = wcs.all_world2pix(ra[i], dec[i], 0) 
    plot_segm(data, xpix=None, ypix=None, name='Candidate {}'.format(i), invert=True, size=100)


