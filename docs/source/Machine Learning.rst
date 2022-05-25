.. _Machine_Learning:

Machine Learning
===========

While the convolutional neural network is the primary engine pyBIA applies for source detection, we explored the utility of other machine learning algorithms as well, of which the random forest was applied as a preliminary filter. Unlike the image classifier, the random forest model we've created takes as input numerous morphological parameters calculated from image moments. These parameters can be calcualated for astrophysical objects using the `catalog module <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_

Given the extended emission features of Lyman-alpha Nebulae, these parameters can be used to differentiate between extended and compact objects which display no diffuse characteristics. Applying the random forest as a preliminary filter ultimately reduces the false-positive rate and optimizes the data requirements of the pipeline. 

Random Forest
-----------

We can create our Random Forest machine learning classifier using the pyBIA `rf_model <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/rf_model/index.html>`_
module:

.. code-block:: python

	from pyBIA import rf_model

	model = rf_model.create(data_x, data_y, impute=False, optimize=False)

If our training data contains invalid values such as NaN or inf, we can impute the missing values using several imputation algorithms. If we impute our data, we must also save the imputer that is fitted to the original training data, so that we can apply it to transform new data if it contains invalid values. We can set impute=True to perform the imputation, but the now the imputer will be the second output.

.. code-block:: python

	model, imputer = rf_model.create(data_x, data_y, impute=True, optimize=False)

We can also set optimize=True, which will perform Bayesian hyperparameter optimization to identify the features that are useful. Doing this will also change the number of outputs, as now there will be a third output, an array containing the indicies of the good features which can be used to index the columns of the data_x array. It is important to index these features every time, as these were the ones that were used to create the final model.

.. code-block:: python

	model, imputer, feats_to_use = rf_model.create(data_x, data_y, impute=True, optimize=True)

By default both of these arguments are set to True. Note that depending on the size of the training data these procedures can take over an hour. 

With our model saved, whether optimized or not, we can use the predict function to pre-process (optional) and predict the class label of unseen data. If the imputation and optimization algorithms are applied, we need to input both the imputer and indices of features to use to ensure we use only the columns that contain information. 

Example:

.. code-block:: python
	
	from pyBIA import rf_model

	model, imputer, feats_to_use = rf_model.create(data_x, data_y, impute=True, optimize=True)

	#Prediction of new, unseen data
	prediction = rf_model.predict(new_data, model, imputer, feats_to_use)

Example
==================

We can load the diffuse_catalog and other_catalog files and create a Random Forest classifier as such:

.. code-block:: python
	
	import pandas
	import numpy as np
	from pyBIA import rf_model

	blob = pandas.read_csv('diffuse_catalog')
	other = pandas.read_csv('other_catalog')
	cols = other.columns.values[8:] #Remove columns that don't include morphological features

	blob = blob[cols]
	other = other[cols]

	mask = np.where(other.area != -999)[0] #-999 are saved when source is a non-detection
	other = other.iloc[mask]

	#Index a random number of OTHER objects, equal to the size of the blob sample
	rand_inx = [int(i) for i in random.sample(range(0, len(mask)), len(blob))] 
	other = other.iloc[rand_inx]

	#Create training data array 
	data_x = np.concatenate((blob, other))

	#Create class label array
	labels_blob = np.array(['DIFFUSE']*len(blob_train))
	labels_other = np.array(['OTHER']*len(other_train))
	data_y = np.r_[labels_blob, labels_other]

	model, imputer, feats_to_use = rf_model.create(data_x, data_y)

Finally, we can predict using our optimized model:

.. code-block:: python

	prediction = rf_model.predict(new_data, model=model, imputer=imputer, feats_to_use=feats_to_use)


Assessing RF Performance
-----------

Using the model created above, we generate both a confusion matrix and a ROC curve.

.. code-block:: python

	from pyBIA import rf_model

	#Confusion Matrix
	rf_model.plot_conf_matrix(classifier=model, data_x, data_y, classes=["DIFFUSE","OTHER"])

	#ROC Curve
	rf_model.plot_roc_curve(classifier=model, data_x, data_y)

For more information refer to the `module documentation <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/rf_model.html>`_.


Convolutional Neural Network
-----------









