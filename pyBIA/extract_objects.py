import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

#216.3969016 32.8258449
hdu = fits.open('/Users/daniel/Desktop/NDWFS_Tiles/Bw_FITS/NDWFSJ1426p3236_Bw_03_fix.fits')
zp = hdu[0].header['MAGZERO']
data = hdu[0].data
wcsobj= wcs.WCS(header = hdu[0].header)

mean, median, std = sigma_clipped_stats(data, sigma=3.0) 

daofind = DAOStarFinder(fwhm=3.0, threshold=2.*std)  
sources = daofind(data - median)  
for col in sources.colnames:  
    sources[col].info.format = '%.8g'  # for consistent table output
print(sources)  

ra, dec = wcsobj.all_pix2world(1000, 1000, 0) 

positions = np.transpose((x_pix, y_pix))
apertures = CircularAperture(positions, r=15.)

annulus_apertures = CircularAnnulus(positions, r_in=20., r_out=35.)
annulus_masks = annulus_apertures.to_mask(method='center')
annulus_data = annulus_masks.multiply(data)
mask = annulus_masks.data
annulus_data_1d = annulus_data[mask > 0]
#_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
median_bkg = np.median(annulus_data_1d)
phot_table = aperture_photometry(data, apertures, error=error)

photometry = phot_table['aperture_sum'] - (median_bkg * apertures.area)
photometry_err = phot_table['aperture_sum_err']
                            
if photometry > 0:
    magnitude = -2.5*np.log10(photometry[0])+zp
    magnitude_err = (2.5/np.log(10))*(photometry_err[0]/photometry[0])
else:
    magnitude=999
    magnitude_err=999

new_data=fixed_size_subset(data, int(y_pix), int(x_pix), 100)
threshold = detect_threshold(new_data, nsigma=1.)

sigma = 3.0 * gaussian_fwhm_to_sigma   
kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
kernel.normalize()
segm = detect_sources(new_data, threshold, npixels=5, filter_kernel=kernel)

