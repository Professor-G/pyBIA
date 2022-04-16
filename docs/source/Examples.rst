.. _examples:

Examples
========
While we configured pyBIA for astrophysical image filtering, the program modules can be called directly to create any type of image classifier. 

Example 1: Ecological Application
-----------

We can train a CNN to differentiate between different animal feces, and while we can create as many different classes as we wish, pyBIA is currently configured for binary classification, therefore in this example we will create two classes. One class will contain images of Chinese water deer samples, while the other class will be composed of images of all other deer native to the region. 

We will train a classifier using 100x100 images in a single filter, which we will assume are stored as separate text files in two distinct directories. First the images will be loaded from their corresponding folders and stored in 3D arrays:

.. code-block:: python

	import os
	import numpy as np

	water_deer = []
	other_deer = []

	for filename in os.listdir('water_deer/'):
		water_deer.append(np.loadxt(filename))

	for filename in os.listdir('other_deer/'):
		other_deer.append(np.loadtxt(filename))

	water_deer = np.array(water_deer)
	other_deer = np.array(other_deet)

Now that we have two arrays containing our training data, we can load pyBIA.models and create our classifier using the default hyperparameters:

.. code-block:: python

	from pyBIA import models

	model = models.create(water_deer, other_deer)

When training is complete, the CNN model will be saved as an .h5 file so that it can be loaded directly in the future using tensorflow.keras.models.load_model().

With our machine learning model saved, we can now classify any 100x100 image of deer feces as either Chinese water deer or not.

.. code-block:: python

	prediction = models.predict(new_data, model, target='WATER_DEER')

This prediction will either be 'WATER_DEER' or 'OTHER'. Note that we had to set  target='WATER_DEER', as by default if the prediction comes out positive, the output is 'DIFFUSE' as the original goal of detecting diffuse Lyman-alpha emission.

<<<<<<< HEAD
Example 2: Green Bean Galaxies (3 filters)
=======
=======

Example 2: Green Bean Galaxies
-----------
>>>>>>> b2f9b6c3d7be38807ab313d84a065c37f04c2a4e





