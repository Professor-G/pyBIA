.. _Engineering_pyBIA:

Engineering pyBIA
===========
To train pyBIA v1.0 we made use of blue broadband survey data from the 9.3 square degree `Boötes field <https://legacy.noirlab.edu/noao/noaodeep/>`_. There are 27 subfields, a fits file for each of these fields can be downloaded `here <https://legacy.noirlab.edu/noao/noaodeep/DR3/DR3cats/matchedFITS/>`_.

To create pyBIA we did the following:

-  Constructed a master catalog of all objects in the Boötes field.
-  Identified Lyman-alpha blob candidates in the catalog using the results from `Moire et al 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_ and extracted blue broadband images for all blob candidates.
-  Applied data augmentation techniques to artificially increase the number of blob samples, which composed our DIFFUSE training class.
-  Extracted blue broadband images for random objects in all field, this made up our OTHER training class.
-  Used the DIFFUSE and OTHER image data to train a Convolutional Neural Network, modeled after the award-winning `AlexNet <https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_ architecture.

1) Constructing the Master Catalog
-----------
We first downloaded the data for all subfields within the `Boötes survey <https://legacy.noirlab.edu/noao/noaodeep/>`_ -- with these 27 fits files we can use pyBIA to automatically detect sources and create a photometric and morphological catalog, although the NDWFS team included `merged catalogs <https://legacy.noirlab.edu/noao/noaodeep/DR3/DR3cats/matchedFITS/>`_ with their data release. We extracted four items from their merged catalogs, the ra & dec positions of each object, as well as the name of the corresponding subfield and the its NDWFS catalog name. This was saved as a Pandas dataframe.

To create a catalog of morphological parameters using pyBIA, we need to use the catalog module, which takes as inputs the x and y pixel positions of each object; for this reason we load astropy and use their `World Coordinate System implementation <https://docs.astropy.org/en/stable/wcs/index.html>`_ to convert our ra/dec equatorial coordinates to image pixel coordinates.

We start by importing our modules and loading our Pandas dataframe containing the following items, extracted from the NDWFS merged catalogs:  ra // dec // field_name // NDWFS_objname

.. code-block:: python

    import pandas
    import astropy
    import numpy as np

    from pyBIA import catalog

    NDWFS_bootes = pandas.read_csv('ndwfs_bootes') 

Since there are 27 different subfields, we load each one at a time and then create a catalog of only the objects that exist within the subfield. For this reason we append each of the 27 catalogs to an empty frame, after which we can concantenate our frame list and save it as a master catalog.

.. code-block:: python
	
    frame = []		#empty list which will store the catalog of every subfield

    for field_name in np.unique(NDWFS_bootes['field_name']):

    	index = np.argwhere(NDWFS_bootes['field_name'] == field_name) 	#identify objects in this subfield
    	hdu = astropy.io.fits.open(path+field_name)					#load .fits field for this subfield only

		wcsobj = astropy.wcs.WCS(header = hdu[0].header)			#create wcs object for coord conversion

		xpix, ypix = wcsobj.all_world2pix(NDWFS_bootes['ra'], NDWFS_bootes['dec'], 0) #convert ra/dec to xpix/ypix
		
		dataframe = catalog.create(hdu[0].data, x=xpix, y=ypix, name=NDWFS_bootes['NDWFS_objname'], morph_params=True, invert=True, save_file=False)

		frame.append(dataframe)

    pd.concat(frames)						#merge all 27 catalogs into one dataframe
    frames.to_csv('NDWFS_master_catalog') 	#save dataframe

When creating a catalog using pyBIA there are numerous parameters you can control, `see the API reference for the catalog class <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_.

2) DIFFUSE Training Class
-----------
`Moire et al 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_ conducted a systematic search of Lyman-alpha Nebulae in the Boötes field, from which 866 total candidates were selected. We start by 



3) Data Augmentation
-----------


4) OTHER Training Class
-----------


5) Creating and Training pyBIA
-----------

