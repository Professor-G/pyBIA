.. _Engineering_pyBIA:

Engineering pyBIA
===========
To train pyBIA v1.0 we made use of blue broadband survey data from the 9.3 square degree `Boötes field <https://legacy.noirlab.edu/noao/noaodeep/>`_. There are 27 subfields, a fits file for each of these fields can be downloaded `here <https://legacy.noirlab.edu/noao/noaodeep/DR3/DR3cats/singleFITS/>`_.

To create pyBIA we did the following:

-  Constructed a master catalog of all objects in the Boötes field.
-  Identified Lyman-alpha blob candidates in the catalog using the results from `Moire et al 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_ and extracted blue broadband images for all blob candidates.
-  Applied data augmentation techniques to artificially increase the number of blob samples, which composed our DIFFUSE training class.
-  Extracted blue broadband images for random objects in all field, this made up our OTHER training class.
-  Used the DIFFUSE and OTHER image data to train a Convolutional Neural Network, modeled after the award-winning `AlexNet <https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf>`_ architecture.

We outline below how we created pyBIA, and the associated model which has been saved and included in the installation.


1) Constructing the Catalog
-----------
We first downloaded the data for all subfields within the `Boötes survey <https://legacy.noirlab.edu/noao/noaodeep/>`_ -- with these 27 fits files we can use pyBIA to automatically detect sources and create a photometric and morphological catalog, although the NDWFS team included `merged catalogs <https://legacy.noirlab.edu/noao/noaodeep/DR3/DR3cats/matchedFITS/>`_ with their data release. We extracted four items from their merged catalogs: the ra & dec positions of each detected source, the name of the corresponding subfield, and its object name. 

To create a catalog we can use the pyBIA.catalog module, which takes as optional inputs the x and y pixel positions of each object (if no positions are entered then DAOFINDER is applied to detect sources). Since we need pixel positions we need to load astropy and use their `World Coordinate System implementation <https://docs.astropy.org/en/stable/wcs/index.html>`_ to convert our ra/dec equatorial coordinates to image pixel coordinates.

We start by importing our modules and loading the file csv file 'ndwfs_bootes', which contains the following items extracted from the NDWFS merged catalogs:  ra // dec // field_name // NDWFS_objname. 

.. code-block:: python
	
    import pandas
    import numpy as np
    import astropy
    from astropy.io.fits import open, getdata

    from pyBIA import catalog

    ndwfs_bootes = pandas.read_csv('ndwfs_bootes') 

Since there are 27 different subfields, we load each one at a time and then create a catalog of only the objects that exist within the subfield. For this reason we create 27 different catalogs and append each to an empty frame, after which we can concantenate our frame list and save it as a master catalog.

.. code-block:: python
	
    frame = []		#empty frame which will store the catalog of every subfield

    for field_name in np.unique(ndwfs_bootes['field_name']):

    	index = np.where(ndwfs_bootes['field_name'] == field_name)[0]  #identify objects in this subfield
    	hdu = open(field_name+'.fits')	 #load .fits field for this subfield only

    	wcsobj = astropy.wcs.WCS(header = hdu[0].header)  #create wcs object for coord conversion
    	xpix, ypix = wcsobj.all_world2pix(ndwfs_bootes['ra'][index], ndwfs_bootes['dec'][index], 0) #convert ra/dec to xpix/ypix

    	cat = catalog.Catalog(data=hdu[0].data, x=xpix, y=ypix, obj_name=ndwfs_bootes['obj_name'][index], field_name=ndwfs_bootes['field_name'][index], flag=np.ones(len(index)), invert=True)
    	cat.create(save_file=False)
    	frame.append(cat.cat)

    frame = pandas.concat(frame, axis=0, join='inner') #merge all 27 frames into one
    frame.to_csv('NDWFS_master_catalog', chunksize=1000) #save as 'NDWFS_master_catalog' 

The NDWFS Bootes field catalog contains a total of 2509039 detected objects. When creating a catalog using pyBIA there are numerous optional parameters that can be contolled, `see the API reference for the catalog class <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_. These features can be used to train a machine learning model, which is why we've included a flag parameter, in which we can input an array containing labels or flags for each object. In the above example, we flagged every object with a value of one, which is what we label any astrophysical object that is not a Lyman-alpha blob. This flag input can also contain an array of strings, which could correspond to actual class labels, e.g. 'GALAXY' or 'STAR'

2) DIFFUSE Training Class
-----------
`Moire et al 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_ conducted a systematic search for Lyman-alpha Nebulae in the Boötes field, from which 866 total candidates were selected after visual inspection. From this sample, 85 were of particular interest as they were within the (bluer) color space of a handful of confirmed Lyman-alpha Nebulae. Their bluer color could indicate stronger Lyman-alpha emission at z > 2, as at :math:`\lambda` ~ 1210 Angstroms, this hydrogen emission would be redshifted to blue when observed on Earth. Only about a third of these 85 candidates have been followed up, and to-date only 5 of these sources have been sprectoscopically confirmed as true Lyman-alpha nebulae. 

The entire sample of 866 objects display morphologies and features which are characteristic of diffuse emission, as such we can begin by extracting these 866 sources from our master catalog. These objects will serve as our initial training sample of diffuse nebulae. We will begin by loading the NDWFS object names of these 866 candidates which we have saved as a file titled 'obj_names_866'. Each object in the survey has a unique name, therefore this can be used to index the master catalog.

.. code-block:: python
	
    master_catalog = pandas.read_csv('NDWFS_master_catalog')
    obj_names_866 = np.loadtxt('obj_names_866', dtype=str)

    index_866 = []

    for obj_name in obj_names_866:

    	index = np.where(master_catalog['obj_name'] == obj_name)[0]

    	#In case there are multiple detections, select the brightest one or flux/flux_err if error argument is input
    	if len(index) > 1:
    		index = index[np.argmax(master_catalog.iloc[index]['flux'])]

    	index_866.append(index)

When we initially created the catalog, we set the 'flag' column to 1 for all objects, but now that we have the indices of the 866 blob candidates, we can set the 'flag' column to 0 for these entries, which we will interpret to mean DIFFUSE. The master catalog can now be separated into a diffuse_catalog containing only these 866 candidates, and an other_catalog with everything else.

.. code-block:: python

	master_catalog['flag'].iloc[index_866] = 0
	other_index = np.where(master_catalog['flag'] == 1)[0]

	diffuse_catalog = master_catalog.iloc[index_866]
	other_catalog = master_catalog.iloc[other_index]

Finally, we will extract 2D arrays of size 100x100, centered around the positions of each of the 866 diffuse objects. We need these images to train the CNN. As was done when creating the catalog, we will loop over all 27 subfields, find the objects in each one, crop out the subarray, and append the images to a list. We can crop out the image of each object using the crop_image function in pyBIA.data_processing:

.. code-block:: python
	
    from pyBIA import data_processing

    diffuse_images = []

    for field_name in np.unique(diffuse_catalog['field_name']):
    	index = np.where(diffuse_catalog['field_name'] == field_name)[0]  #identify objects in this subfield
    	hdu = open('/Users/daniel/Desktop/NDWFS_Tiles/Bw_FITS/'+field_name+'_Bw_03_fix.fits')   #load .fits field for this subfield only
    	data = hdu[0].data
    	for i in range(len(index)): #Crop out objects

    		xpix = diffuse_catalog.xpix.iloc[index[i]]
    		ypix = diffuse_catalog.ypix.iloc[index[i]]
    		image = data_processing.crop_image(data, x=np.array(xpix), y=np.array(ypix), size=100, invert=True)
    		diffuse_images.append(image)

    diffuse_images = np.array(diffuse_images)

The diffuse_images array now contains image data for our 'DIFFUSE' training class (flag=0), but a training class of 866 objects is very small. AlexNet, the convolutional neural network pyBIA is modeled after, used ~1.3 million images for training. Since Lyman-alpha nebulae are rare we don't have a large sample of these objects, as such, we must perform data augmentation techniques to inflate our 'DIFFUSE' training bag, after which we can randomly select a similar number of other objects to compose our 'OTHER' training class. 

3) Data Augmentation
-----------
We want to apply modification techniques to our images of DIFFUSE objects in ways that will not alter the integrity of the morphological characteristics, so data augmentation methods that include image zoom and cropping, as well as pixel alterations, should not be applied in this context. We adopted the following combination of data augmentation techniques:

-  Horizontal Shift
-  Vertical Shift 
-  Horizontal Flip
-  Vertical Flip
-  Rotation

Each time an augmented image is created, the shifts, flips, and rotation parameters are chosen at random as per the specified bounds. It's important to note that image shifts and rotations do end up altering the original image, as the shifted and distorted areas require filling either by extrapolation or by setting the pixels to a constant value -- it is for this reason that we extracted the images of our 866 DIFFUSE objects as 100x100 pixels. We will first perform data augmentation, after which we will resize the image to 50x50. This ensures that any filling that occurs on the outer boundaries because of shifts or rotations end up being cropped away.

To perform data augmentation, we can use pyBIA's data_augmentation module, we just need to input how many augmented images per original sample we will create, and the specified bounds of the augmentations. For help please see the `data augmentation documentation <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/data_augmentation/index.html>`_. We decided to create 100 augmented images per object, enable horizontal/vertical flips and full rotation, and allow for horizontal and vertical shifts of 5 pixels in either direction. Each augmented image will be created by randomly sampling from random uniform distributions.

.. code-block:: python

	from pyBIA.data_augmentation import augmentation

	diffuse_training = augmentation(diffuse_images, batch=10, width_shift=5, height_shift=5, horizontal=True, vertical=True, rotation=360)

By default the augmentation function will resize the image to 50x50 after performing the data augmentation, but this resizing can be controlled with the image_size argument. 

The diffuse_training variable is a 3D array containing 866*100=86600 augmented images -- this array will be our 'DIFFUSE' training bag. We can now extract a similar number of other objects to compose our 'OTHER' training bag. This is one power of data augmentation: by inflaating the size of the data-deprived class, you can include more data of the other classes for which there are more samples.

4) OTHER Training Class
-----------
It is important to avoid class imbalance when training machine learning algorithms. The sizes of each class should be relatively the same so as to avoid fitting issues; therefore we're going to extract 50x50 images of 86600 random sources, chosen from the other_catalog:


.. code-block:: python
	
    rand_index = random.sample(range(len(other_catalog)), 86600) #random index
    other_catalog = other_catalog.iloc[rand_index]

    other_images = []

    for field_name in np.unique(other_catalog['field_name'].iloc[rand_index]):

    	index = np.where(other_catalog['field_name'] == field_name)[0]  #identify objects in this subfield
    	hdu = open(field_name+'.fits')	 #load .fits field for this subfield only
    	data = hdu[0].data

    	for i in range(len(index)): #Crop out objects

    		xpix = other_catalog.xpix.iloc[index[i]]
    		ypix = other_catalog.ypix.iloc[index[i]]
    		image = data_processing.crop_image(data, x=np.array(xpix), y=np.array(ypix), size=50, invert=True)
    		other_images.append(image)

    other_training = np.array(other_images)

With these two 3D arrays containing 86600 samples eah (diffuse_training & other_training), we can create a binary classifier.

5) Training pyBIA
-----------
To properly evaluate classification performance, it is imperative that we create a validation dataset that will evaluated at the end of every training epoch. We will separate 10% of the data for validation by shuffling the two training arrays and then selecting the first 10 percent of the array as our validation data.

.. code-block:: python

	import random

	random.shuffle(diffuse_training)
	random.shuffle(other_training)

Since we have 86600 samples in each array, we will index the first 8660 to be the validation data, which we can construct using the data_processing.process_class() function. This function takes as input a 3D array containing images of a single class, all categorized with the same label. In our case the label 0 corresponds to DIFFUSE, and 1 to OTHER; therefore we need to create two validation sets, one for DIFFUSE and one for OTHER, after which we'll combine to form one validation set

.. code-block:: python

	val_X1, val_Y1 = data_processing.process_class(diffuse_training[:866], label=0, min_pixel=638, max_pixel=1500)
	val_X2, val_Y2 = data_processing.process_class(other_training[:866], label=1, min_pixel=638, max_pixel=1500)

	val_X = np.r_[val_X1, val_X2]
	val_Y = np.r_[val_Y1, val_Y2]

The process_class function will output two arrays, the reshaped image data and the appropriately shaped labels. Both of these arrays are reshaped in preparation for the training. 

IMPORTANT: When doing image classification it is important to normalize the images so as to avoid exploding gradients. We applied min-max normalization, where min_pixel is the average background count of the data (or entire survey); in our case we set the min to be 638, the 0.01 quantile of the Boötes field. The max_pixel value is set to 1500, we set this value because Lyman-alpha nebulae are diffuse sources and thus we can ignore anything brighter than 1500,  which will result in more robust classification performance.

Since we used the first 10% of the data for validation, the remaining 90% will be used to train the CNN, we will create the CNN model using pyBIA.models.create():

.. code-block:: python

	from pyBIA import cnn_model

	model = cnn_model.create(diffuse_training[8660:], other_training[8660:], val_X=val_X, val_Y=val_Y, min_pixel= 638, max_pixel=1500, filename='Bw_CNN')

When the pyBIA model is trained it will save metric files and an .h5 file containing the Tensorflow model. We did not set any of the parameters in the above example as the default ones are the ones we used, but please note that by default the CNN will train for 1000 epochs, which would take several days to complete. Because of the computation time needed to train the model, a checkpoint file will automatically be saved everytime the performance improves, that way we can resume training should the process be interrupted.

With our model saved we can now classify any object by entering the 50x50 2D arrays, either individually or as a 3D array:

.. code-block:: python
	
	prediction = models.predict(data, model, normalize=True, min_pixel=638, max_pixel=1500)

Trained models for Lyman-alpha blob detection are included in the pyBIA installation and can be loaded directly. For more information on how to run pyBIA modules please see the Example page. 

Machine Learning
-----------

While the convolutional neural network is the primary engine pyBIA applies for source detection, we explored the utility of other machine learning algorithms as well, of which the random forest was applied as a preliminary filter. Unlike the image classifier, the random forest model we've created takes as input numerous morphological parameters calculated from image moments. 

Given the extended emission features of Lyman-alpha Nebulae, these parameters can be used to differentiate between extended and compact objects which display no diffuse characteristics. Applying the random forest as a preliminary filter ultimately reduces the false-positive rate and optimizes the data requirements of the pipeline. 

Details on the machine learning models are available `here <https://pybia.readthedocs.io/en/latest/source/Machine%20Learning.html>`_. 











