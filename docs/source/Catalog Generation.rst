.. _catalog:

Morphological Catalog
=====================

.. admonition:: Under Construction (last updated 2026-01-04)

   This documentation is still being written and may change frequently!

The `catalog <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_ module is the engine of pyBIA’s feature extraction pipeline. It handles source detection, photometry, and the calculation of advanced morphological descriptors necessary for machine learning classification.

By integrating **image segmentation** with **orthonormal moment analysis**, pyBIA converts raw pixel data into a structured feature matrix containing:

* **Photometry:** Fluxes, magnitudes, and photometric errors.
* **Geometric Invariants:** Hu moments and raw image moments.
* **Orthogonal Features:** Legendre moments for uncorrelated shape description.
* **Morphometry:** Standard SourceExtractor-like shape parameters (ellipticity, FWHM, etc.).

Quick Start
-----------

If you have a 2D image array containing many sources, and do not require specific positional extraction, you can generate a catalog immediately using the built-in auto-detection:

.. code-block:: python

    from pyBIA import catalog

    # Instantiate the Catalog class (data is the image)
    cat = catalog.Catalog(data)

    # Run the source-detection and image segmetation routine
    cat.create(save_file=True)

    # The catalog is stored in the ``cat`` class attribute 
    print(cat.cat)

Computed Features
-----------------

The resulting catalog is comprehensive, containing approximately 50 columns per source. These are derived from two internal routines:

1.  **Moments Analysis** (`pyBIA.image_moments`):
    Calculates pixel-intensity weighted moments on the segmented source.

    * **Raw & Central Moments:** (:math:`M_{00}` ... :math:`M_{03}`) describing spatial distribution.
    * **Hu Moments:** Scale, translation, and rotation invariant moments (:math:`Hu_1` ... :math:`Hu_7`).
    * **Legendre Moments:** Orthonormal moments (:math:`L_{00}` ... :math:`L_{21}`) computed up to the 3rd order. Unlike raw moments, these provide an orthogonal representation of the source profile, reducing feature correlation.

2.  **Morphometry** (`photutils.segmentation`):
    Shape, size, and photometric parameters compatible with SourceExtractor definitions.

    * **Shape & Geometry:** Ellipticity, Elongation, Eccentricity, Orientation, Perimeter, and Equivalent Radius.
    * **Covariance & Ellipse:** Covariance Eigenvalues (:math:`\lambda_1, \lambda_2`), Covariance Matrix elements (`covar_sigx2`, `covar_sigy2`, `covar_sigxy`), and SourceExtractor ellipse parameters (`cxx`, `cxy`, `cyy`).
    * **Size & Distribution:** Area (pixels), FWHM (approximate), Semimajor/Semiminor Axis Sigma, and Gini Coefficient.
    * **Indices & Bounds:** Bounding Box coordinates (`xmin`, `xmax`, `ymin`, `ymax`), Max/Min pixel values, and their corresponding pixel indices.
    * **Photometry:** Circular Aperture flux (fixed radius) and photometric errors.

Methodology & Parameters
------------------------

For scientific workflows, precise control over the extraction window and background estimation is required. The `Catalog` class accepts specific coordinates via `x` and `y`, centering the analysis on your targets.

The pipeline follows this logic: **Crop** :math:`\rightarrow` **Background Subtraction** :math:`\rightarrow` **Convolve** :math:`\rightarrow` **Segment** :math:`\rightarrow` **Measure**.

.. list-table:: Key Parameters
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``x``, ``y``
     - None
     - Pixel coordinates of the source center.
   * - ``size``
     - 100
     - Size (in pixels) of the square cutout window to crop around the source.
   * - ``bkg``
     - None
     - Background mode. Set to ``0`` if data is already background-subtracted. If ``None``, local background is estimated via annuli.
   * - ``annulus_in``
     - 20
     - Inner radius of the background estimation annulus.
   * - ``annulus_out``
     - 35
     - Outer radius of the background estimation annulus.
   * - ``aperture``
     - 15
     - Radius of the circular aperture used for flux calculation.
   * - ``nsig``
     - 0.3
     - Detection threshold (sigma above background) for a pixel to be included in the segmentation map.
   * - ``deblend``
     - ``False``
     - If ``True``, attempts to deblend overlapping sources. ``False`` is recommended for capturing environmental features (e.g., proto-clusters).
   * - ``threshold``
     - 10
     - Validation radius. The nearest segmentation patch is only accepted if it falls within this distance (pixels) from the input center.

.. note::
   **Non-Detections:** If no segmentation patch is found within the ``threshold`` radius (or if the source is too faint for ``nsig``), pyBIA flags the source as a non-detection. All morphological features (moments, shape) will be set to **-999**. However, forced aperture photometry (flux/magnitude) will still be recorded.

Tutorial: Building an LSST Catalog
----------------------------------

In this example, we generate a source catalog for a dataset of 20,000 simulated extragalactic sources (10k strong lenses, 10k non-lenses). These sources are simulated in the five bands LSST will observe (*g, r, i, z, y*).

**Data Access**
You can download the sample binary files here:

* `lenses_10k <https://drive.google.com/file/d/1fpr1LIPD08qBkeER0q3hREhdi9g40pDm/view?usp=sharing>`_
* `nonlenses_10k <https://drive.google.com/file/d/1EQK1o0INWbMpVr-2qh_8MLiDNzG6fyVT/view?usp=sharing>`_

**Processing Script**
We will process each band individually, constructing separate catalogs for lenses and non-lenses, and then merging them.

.. code-block:: python
   :linenos:

   import numpy as np 
   import pandas as pd 
   from pyBIA import catalog  

   # Load the images, binary files (41x41 pixels) containing five filters: g,r,i,z,y
   lenses = np.load('lenses_10k.npy')
   nonlenses = np.load('nonlenses_10k.npy')

   # pyBIA catalog parameteres #

   error = None # The corresponding error map, for computing photometric errors 
   xpix = ypix = lenses.shape[-1] // 2 # Relative position (in pixels) of the source centroid, here they are centered about the image cutouts
   bkg = None # None if background subtraction required, else set to 0 if data already bg-subtracted
   aperture = 10 # Aperture radius (in pixels) for the photometry
   annulus_in = 15 # Inner radius (in pixels) of background annulus for local sky estimation
   annulus_out = 50 # Outer radius (in pixels) of background annulus. 
   nsig = 0.3 # The image segmentation detection threshold 
   threshold = 1 # Will plot the closest object within a circular mask of radius 10 (pixels) within the center
   exptime = 1 # Radius (in pixels) around the source center used to validate detection. If no object is found within this region, the source is flagged as a non-detection.
   zp = 27 # The instrumental zeropoint, for computing the apparent magnitudes
   deblend = False # Whether to deblend detected source(s)
   kernel_size = 21 # Gaussian filter kernel size used to convolve the data prior to segmentation
   npixels = 9 # Required number of pixels above the sigma threshold required to detect a source
   connectivity = 8 # Scheme to determine how pixels are grouped into a detected source, either 4 (touch along edges) or 8 (edges and corners)

   # Will process one band at a time, and save each individually
   for i, band in enumerate(['g','r','i','z','y']):

           # To save all of the individual catalogs
           master_catalog = [] 

           # Loop through each individual lens source
           for j in range(len(lenses)):
                   obj_name = j # The object name in the catalog will just be the order as it appears in the data
                   flag = 1 # The positive class label

                   # Instantiate the Catalog class
                   cat = catalog.Catalog(
                           data=lenses[j][i], # First axis selects the individual source, second axis the band 
                           error=error, 
                           x=xpix, 
                           y=ypix, 
                           bkg=bkg, 
                           aperture=aperture,
                           annulus_in=annulus_in,
                           annulus_out=annulus_out,
                           nsig=nsig, 
                           threshold=threshold,
                           exptime=exptime,
                           zp=zp,
                           deblend=deblend, 
                           kernel_size=kernel_size,
                           npixels=npixels,
                           connectivity=connectivity,
                           obj_name=obj_name,
                           flag=flag
                           )

                   # Create the catalog, will be stored as the `cat` class attribute.
                   cat.create(save_file=False) 

                   # Append the created `cat` attribute to the master list 
                   master_catalog.append(cat.cat)

           ##
           ## Now repeat for the non-lenses ##
           ##

           # Loop through each individual non-lens source
           for j in range(len(nonlenses)):
                   obj_name = len(lenses) + j # The object name in the catalog will just be the order as it appears in the data + how many lenses there are already
                   flag = 0 # The negative class label

                   # Instantiate the Catalog class
                   cat = catalog.Catalog(
                           data=nonlenses[j][i], # First axis selects the individual source, second axis the band 
                           error=error, 
                           x=xpix, 
                           y=ypix, 
                           bkg=bkg, 
                           aperture=aperture,
                           annulus_in=annulus_in,
                           annulus_out=annulus_out,
                           nsig=nsig, 
                           threshold=threshold,
                           exptime=exptime,
                           zp=zp,
                           deblend=deblend, 
                           kernel_size=kernel_size,
                           npixels=npixels,
                           connectivity=connectivity,
                           obj_name=obj_name,
                           flag=flag
                           )

                   # Create the catalog, will be stored as the `cat` class attribute.
                   cat.create(save_file=False) 

                   # Append the created `cat` attribute to the master list 
                   master_catalog.append(cat.cat)


           # Now Merge all individual catalogs into one master dataframe and save
           df = pd.concat(master_catalog, ignore_index=True)
           df.to_csv(f'segm_catalog_{band}_band.csv', index=False)

The five catalogs generated above are available for download here:

* `segm_catalog_g_band <https://drive.google.com/file/d/11IE0XTBl-xI6VtL_6objv5G9IKKt1ZB6/view?usp=sharing>`_
* `segm_catalog_r_band <https://drive.google.com/file/d/1--JoD2hB_sBb8AwtW4DZVllbQFolpe6v/view?usp=sharing>`_
* `segm_catalog_i_band <https://drive.google.com/file/d/189rstaGZrSBT679SK2HVSBDMTslJPAJw/view?usp=sharing>`_
* `segm_catalog_z_band <https://drive.google.com/file/d/1rqRXs05nPeKB9qQUtEw0BAipTQ9yZ3Y6/view?usp=sharing>`_
* `segm_catalog_y_band <https://drive.google.com/file/d/1Zp0n1xed_EcUsgC3Y4G7GXgrUQOStFCH/view?usp=sharing>`_

These catalogs will be merged and used to train a binary classifier using the `ensemble_model <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/ensemble_model/index.html>`_ module. This example is provided in the `Supervised Learning Algorithms <https://pybia.readthedocs.io/en/latest/source/Supervised%20Learning%20Algorithms.html>`_ page. 

**NOTE:** The catalog module also provides a standalone function to plot individual sources and the corresponding image segmentation patches given some set of parameter(s). The `plot_objects_segmentation <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html#pyBIA.catalog.plot_objects_segmentation>`_ function allows users to inspect the segmentation masks overlaid on the source. As the source morphological features are dependent on the resulting segmentation, it is important to ensure the generated patches are truly representative of the source morphology. This function allows users to input up to four segmentation detection thresholds (``sigma_values``), so as to visualize how different values affect the resulting source extent.

In the example below, we inspect a lens where only the *i-band* yields a positive detection at the strictest threshold (:math:`\sigma=5.0`). The other four filters at this detection level would be non-detections and the corresponding morphological features would thus be cataloged with -999 values. 

.. code-block:: python

   import numpy as np
   from pyBIA import catalog

   # Load the lenses 
   lens = np.load('lenses_10k.npy')

   # Plotting parameters
   median_bkg = None # Whether to subtract the background (set to None if background subtraction required)
   pix_conversion = 5.8 # Survey pixel-per-arcsecond (for setting the axes)
   crop_size = None # Will crop the image to be of this size, otherwise set to None
   xpix = ypix = lens.shape[2] // 2 # Cropped image will be centered about these coords, if not cropping set to None
   r_in = 15 # Inner radius (in pixels) of background annulus for local sky estimation
   r_out = 50 # Outer radius (in pixels) of background annulus. 

   # Figure parameters
   fig_title = r'Example Lens' # Figure suptitle
   sup_titles = [r'$g$', r'$r$', r'$i$', r'$z$', r'$y$'] # Title(s) above each individual panel
   cmap = 'viridis' # Colormap to use when displaying input image, the segmentation patches always use binary

   # Segm detection parameters
   sigma_vals = [0.3, 1.0, 3.0, 5.0] # The detection threshold(s) to apply
   deblend = False # Whether to deblend detected sources
   kernel_size = 21 # Gaussian filter kernel size used to convolve the data prior to segmentation
   npixels = 9 # Required number of pixels above the sigma threshold required to detect a source
   connectivity = 8 # Scheme to determine how pixels are grouped into a detected source, either 4 (touch along edges) or 8 (edges and corners)
   threshold = 0 # Will plot the closest object within a circular mask of radius 10 (pixels) within the center
   savefig = True # Whether to save the figure, it False it will show instead
   savepath = 'segm_example_lens.png' # Path (and/or filename) to save in/as

   i = 4 # Will plot the fifth source in the array

   # This function takes in up to 5 images, and plots the detection thresholds (up to 4 thresholds allowed)
   catalog.plot_objects_segmentation(
      lens[i][0],
      lens[i][1],
      lens[i][2],
      lens[i][3],
      lens[i][4],
      pix_conversion=pix_conversion,
      sigma_values=sigma_vals,
      deblend=deblend,
      kernel_size=kernel_size,
      npixels=npixels,
      connectivity=connectivity,
      threshold=threshold,
      titles=sup_titles,
      suptitle=fig_title,
      cmap=cmap,
      xpix=xpix,
      ypix=ypix,
      size=crop_size,
      median_bkg=median_bkg,
      savefig=savefig,
      r_in=15,
      r_out=50,
      savepath=savepath
      )

.. figure:: _static/segm_example_lens.png
    :align: center
    :alt: Segmentation Example
    :width: 800px

    Visualization of segmentation maps across 5 bands. The binary masks illustrate the detected morphology at increasing sigma thresholds.