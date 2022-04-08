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

We outline below how we created pyBIA, and the associated model which has been saved and included in the installation.


1) Constructing the Catalog
-----------
We first downloaded the data for all subfields within the `Boötes survey <https://legacy.noirlab.edu/noao/noaodeep/>`_ -- with these 27 fits files we can use pyBIA to automatically detect sources and create a photometric and morphological catalog, although the NDWFS team included `merged catalogs <https://legacy.noirlab.edu/noao/noaodeep/DR3/DR3cats/matchedFITS/>`_ with their data release. We extracted four items from their merged catalogs: the ra & dec positions of each detected source, the name of the corresponding subfield, as well as its NDWFS object name. This was saved as a Pandas dataframe.

To create a catalog we can use the pyBIA.catalog module, which takes as optional inputs the x and y pixel positions of each object (if no positions are entered then DAOFINDER is applied to detect sources). Since we need pixel positions we need to load astropy and use their `World Coordinate System implementation <https://docs.astropy.org/en/stable/wcs/index.html>`_ to convert our ra/dec equatorial coordinates to image pixel coordinates.

We start by importing our modules and loading our Pandas dataframe containing the following items, extracted from the NDWFS merged catalogs:  ra // dec // field_name // NDWFS_objname. 

.. code-block:: python

    import pandas
    import astropy
    import numpy as np

    from pyBIA import catalog

    ndwfs_bootes = pandas.read_csv('ndwfs_bootes') 

Since there are 27 different subfields, we load each one at a time and then create a catalog of only the objects that exist within the subfield. For this reason we create 27 different catalogs and append each to an empty frame, after which we can concantenate our frame list and save it as a master catalog.

.. code-block:: python
	
    frame = []		#empty list which will store the catalog of every subfield

    for field_name in np.unique(ndwfs_bootes['field_name']):

    	index = np.argwhere(ndwfs_bootes['field_name'] == field_name)  #identify objects in this subfield
    	hdu = astropy.io.fits.open(path+field_name)	 #load .fits field for this subfield only

		wcsobj = astropy.wcs.WCS(header = hdu[0].header)  #create wcs object for coord conversion
		xpix, ypix = wcsobj.all_world2pix(ndwfs_bootes['ra'][index], ndwfs_bootes['dec'][index], 0) #convert ra/dec to xpix/ypix

		cat = catalog.create(data=hdu[0].data, x=xpix, y=ypix, name=ndwfs_bootes['NDWFS_objname'][index], field_name=ndwfs_bootes['field_name'][index], flag=np.ones(len(index)), invert=True, save_file=False)
		frame.append(cat)

    pd.concat(frames) #merge all 27 catalogs into one dataframe
    frames.to_csv('NDWFS_master_catalog') 	#save dataframe as 'NDWFS_master_catalog'

When creating a catalog using pyBIA there are numerous parameters you can control, `see the API reference for the catalog class <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_. These features can be used to train a machine learning model, which is why we've included a flag parameter, in which we can input an array containing labels or flags for each object. In the above example, we flagged every object with a value of one, which is what we label any astrophysical object that is not a Lyman-alpha blob. This flag input can also contain an array of strings, which could correspond to actual class labels, e.g. 'GALAXY', 'STAR'

2) DIFFUSE Training Class
-----------
`Moire et al 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_ conducted a systematic search for Lyman-alpha Nebulae in the Boötes field, from which 866 total candidates were selected after visual inspection. From this sample, 85 had a larger (B-R), which could indicate stronger Lyman-alpha emission at z > 2. Only about a third of these 85 candidates have been followed up, and to-date only 5 of these sources have been sprectoscopically confirmed as true Lyman-alpha nebulae. 

The entire sample of 866 objects display morphologies and features which are characteristic of diffuse emission, as such we can begin by extracting these 866 sources from our master catalog. These objects will serve as our initial training sample of diffuse nebulae. We will begin by loading the NDWFS object names of these 866 candidates which we have saved as a file titled 'obj_names_866'. Each object in the survey has a unique name, therefore this can be used to index the master catalog.

.. code-block:: python

	master_catalog = pandas.read_csv('NDWFS_master_catalog')
	obj_names_866 = np.loadtxt('obj_names_866', dtype=str)

	866_index = []

	for i in range(len(obj_names_866)):
		index = np.argwhere(master_catalog['name'] == obj_names_866[i])
		866_index.append(int(index))

	866_index = np.array(866_index)

When we initially created the catalog, we set the 'flag' to 1 for all objects, but now that we have the indices of the 866 blob candidates, we can set the 'flag' column to 0 for these entries, which we will interpret to mean DIFFUSE. For simplicity, we will break up our master catalog into a diffuse_catalog containing only these 866 candidates, and an other_catalog with everything else.

.. code-block:: python

	diffuse_catalog = master_catalog[866_index]
	diffuse_catalog['flag'] = 0

	other_index = np.argwhere(master_catalog['flag'] == 1)
	other_catalog = master_catalog[other_index]

Finally, we will extract 2D arrays of size 100x100, centered around the positions of each of the 866 diffuse objects. We need these images to train the CNN. As was done when creating the catalog, we will loop over all 27 subfields, find the objects in each one, crop out the subarray, and append the images to a list. We can crop out the image of each object using the crop_image function in pyBIA.data_processing:

.. code-block:: python
	
	from pyBIA import data_processing

	diffuse_images = []

	for field_name in np.unique(diffuse_catalog['field_name']):

    	index = np.argwhere(diffuse_catalog['field_name'] == field_name)  #identify objects in this subfield
    	hdu = astropy.io.fits.open(path+field_name)	 #load .fits field for this subfield only
    	data = hdu[0].data

    	for i in range(len(index)):
    		image = crop_image(data, x=diffuse_catalog['xpix'], y=diffuse_catalog['ypix'], size=100, invert=True)
    		diffuse_images.append(image)

    diffuse_images = np.array(diffuse_images)

The diffuse_images array now contains data for our 'DIFFUSE' training class (flag=0), but 866 samples is very small. AlexNet, the convolutional neural network pyBIA is modeled after, used ~1.3 million images for training. Since Lyman-alpha nebulae are rare we don't have a large sample of these phenomena, as such, we must perform data augmentation techniques to inflate our 'DIFFUSE' training bag, after which we can randomly select a similar number of other objects to compose our 'OTHER' training class. 

3) Data Augmentation
-----------
We want to apply modification techniques to our images of DIFFUSE objects in ways that will not alter the integrity of the morphological features, so data augmentation methods that include image zoom and cropping, as well as pixel alterations, should not be applied in this context. We adopted the following combination of data augmentation techniques:

-  Horizontal Shift
-  Vertical Shift 
-  Horizontal Flip
-  Vertical Flip
-  Rotation

Each time an augmented image is created, the shifts, flips, and rotation parameters are chosen at random as per the specified bounds. It's important to note that image shifts and rotations do end up altering the original image, as the shifted and distorted areas require filling either by extrapolation or by setting the pixels to a constant value -- it is for this reason that we extracted the images of our 866 DIFFUSE objects as 100x100 pixels. We will first perform data augmentation, after which we will resize the image to 50x50. This ensures that any filling that occurs because of shifts or rotations exist on the outer boundaries of the image which end up being cropped away.

To perform data augmentation, we can use pyBIA's data_augmentation model, we just need to input how many augmented images per original image we will create, and the specified bounds of the augmentations. For help please see the `augmentation documentation <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/data_augmentation/index.html>`_. We decided to create 100 augmented images per original sample, enable horizontal/vertical flips and full rotation, and allow for horizontal and vertical shifts of 5 pixels in either direction. Each augmented image will be created by randomly sampling from the distributions.

.. code-block:: python

	from pyBIA import data_augmentation

	diffuse_training = augmentation(diffuse_images, batch=100, width_shift=5, height_shift=5, horizontal=True, vertical=True, rotation=360)

By default the augmentation function will resize the image to 50x50 after performing the data augmentation, but this resizing can be controlled with the image_size argument. 

The diffuse_training variable is a 3D array containing 866*100=86600 augmented images -- this array will be our 'DIFFUSE' training bag. We can now extract a similar number of other objects to compose our 'OTHER' training bag. 

4) OTHER Training Class
-----------
It is important to avoid class imbalance when training machine learning algorithms. The sizes of each class should be relatively the same so as to avoid fitting issues; therefore we're going to extract 50x50 images of 86600 random sources, chosen from the other_catalog:

.. code-block:: python

	index = random.sample(range(len(other_catalog)), 86600) #random index

	other_images = []

	for field_name in np.unique(other_catalog['field_name']):

    	index = np.argwhere(other_catalog['field_name'] == field_name)  #identify objects in this subfield
    	hdu = astropy.io.fits.open(path+field_name)	 #load .fits field for this subfield only
    	data = hdu[0].data

    	for i in range(len(index)):
    		image = crop_image(data, x=other_catalog['xpix'], y=other_catalog['ypix'], size=100, invert=True)
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

	val_X1, val_Y1 = process_class(diffuse_training[:8660], label=0, min_pixel=638, max_pixel=1500)
	val_X2, val_Y2 = process_class(other_training[:8660], label=1, max_pixel=1500)

	val_X = np.r_[val_X1, val_X2]
	val_Y = np.r_[val_Y1, val_Y2]

The process_class function will output two arrays, the reshaped image data and the appropriately shaped labels. Both of these arrays are reshaped in preparation for the training. 

IMPORTANT: When doing image classification it is imperative that we normalize our images so as to avoid exploding gradients. We applied min-max normalization, where min_pixel is the average background count of the data (or entire survey); in our case we set the min to be 638, the 0.01 quantile of the Boötes field. The max_pixel value is set to 1500, we set this value because Lyman-alpha nebulae are diffuse sources and thus we can ignore anything brighter than 1500,  which will result in more robust classification performance.

Since we used the first 10% of the data for validation, the remaining 90% will be used to train the CNN, we will create the CNN model using pyBIA.models.pyBIA_model():

.. code-block:: python

	model = pyBIA_model(blob_train[8660:], other_train[8660:], validation_X=val_X, validation_Y=val_Y, min_pixel= 638, max_pixel=1500, filename='Bw_CNN')

When the pyBIA model is trained it will save metric files and an .h5 file containing the Tensorflow model. We did not set any of the parameters in the above example as the default ones are the ones we used, but please note that by default the CNN will train for 1000 epochs, which would take several days to complete. Because of the computation time needed to train the model, a checkpoint file will automatically be saved everytime the performance improves, that way we can resume training should the process be interrupted.

With our model saved we can now classify any object by entering the 50x50 2D arrays, either individually or as a 3D array:

.. code-block::python
	
	prediction = models.predict(data, model, normalize=True, min_pixel=638, max_pixel=1500)

In practice we don't need to create models from scratch, as trained models are included in the pyBIA installation and can be loaded directly. For more information see the Example page. 


