.. _Machine_Learning:

Machine Learning
===========

While the convolutional neural network is the primary engine pyBIA applies for source detection, we explored the utility of other machine learning algorithms as well, of which Extreme Gradient Boosting was applied as a preliminary filter. Unlike the image classifier, the ensemble model we've created takes as input numerous morphological parameters calculated from image moments. These parameters can be calcualated for astrophysical objects using the `catalog module <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_

Given the diffuse emission features of Lyman-alpha Nebulae, these parameters can be used to differentiate between extended and compact objects which display no diffuse characteristics. Applying XGB as a preliminary filter ultimately reduces the false-positive rate and optimizes the data requirements of the pipeline. 

Ensemble Model
-----------

We can create an XGB machine learning classifier using the pyBIA `ensemble_models <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/rf_model/index.html>`_
module. Unless turned off, when creating the model three optimization procedures will automatically run, in the following order:

-  Missing values (NaN) will be imputed using the `sklearn implementation of the k Nearest Neighbors imputation algorithm <https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html>`_. The imputer will be saved so that it can be applied to transform new, unseen data, serving as a workaround for the issue of missing data values. 

-  The features that contain information will be selected using `BorutaShap <https://zenodo.org/record/4247618>`_, a procedure based off of the Boruta algorithm developed by `Kursa and Rudnicki 2011 <https://arxiv.org/pdf/1106.5112.pdf>`_. This new method improves upon the original approach by coupling the Boruta algorithm's probabilistic approach to feature selection with `Shapley Values <https://christophm.github.io/interpretable-ml-book/shapley.html>`_. While bagging algorithms like the Random Forest are robust to irrelevant features, computation-wise, it is imperative that we compute only the features that are helpful.

-  Finally, the model hyperparameters will be optimized using the hyperparameter optimization software `Optuna <https://optuna.org/>`_, developed by `Akiba et al 2019 <https://arxiv.org/abs/1907.10902>`_. The default sampler Optuna employs is the Tree Parzen Estimator, a Bayesian optimization approach that effectively reduces the error by narrowing the search space according to the performance of previous iterations, therefore in principle it is best to increase the number of trials to perform.

These methods are enabled by default, but can be turned off when creating our classifier with the impute and optimize arguments:

.. code-block:: python

	from pyBIA import ensemble_model

	model = ensemble_model.classifier(data_x, data_y, clf='xgb', impute=False, optimize=False)
	model.create()

If our training data contains invalid values such as NaN or inf, we can impute the missing values using several imputation algorithms. If we impute our data, the imputer is stored as object attribute as it will be needed to transform new data if it contains invalid values. We can set impute=True to perform the imputation:

.. code-block:: python

	model = ensemble_model.classifier(data_x, data_y, clf='xgb', impute=True, optimize=False)
	model.create()

We can also set optimize=True, which will perform Bayesian hyperparameter optimization to identify the features that are useful. If we do this the attribute model.feats_to_use will contain an array of indices which with to index the feature columns.

.. code-block:: python

	model = ensemble_model.classifier(data_x, data_y, clf='xgb', impute=True, optimize=True)
	model.create()

To avoid overfitting during the optimization procedure, 3-fold cross-validation is performed to assess performance at the end of each trial, therefore the hyperparameter optimization can take a long time depending on the size of the training set and the algorithm being optimized. 

Note that pyBIA currently supports three machine learning algorithms: Random Forest, Extreme Gradient Boosting, and Neural Network. While clf='rf' for Random Forest is the default input, we can also set this to 'xgb' or 'nn'. Since neural networks require more tuning to properly identify the optimal number of layers and neurons, it is recommended to set n_iter to at least 100, as by default only 25 trials are performed when optimizing the hyperparameters:

.. code-block:: python

   model = ensemble_model.classifier(data_x, data_y, clf='nn', n_iter=100)
   model.create()

The Random Forest was the first algorithm we explored, which yielded accuracy shy of that provided by the XGB implementation. To make a Random Forest classifier we just set the `clf` argument to 'rf':

.. code-block:: python

   model = ensemble_model.classifier(data_x, data_y, clf='rf')
   model.create()

`For details please refer to the function documentation <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/ensemble_models/index.html#pyBIA.ensemble_models.create>`_.

With our model saved, whether optimized or not, we can use the predict method to predict the class label of unseen data. 

Example:

.. code-block:: python

	prediction = model.predict(new_data)

Visualizations
-----------

A confusion matrix, ROC curve, and/or t-SNE projection can be displayed via the corresponding model attributes:

.. code-block:: python

	model.plot_conf_matrix()
	model.plot_roc_curve()
	mode.plot_tsne()

If optimizated, `model.optimization_results` and `model.feature_history` will contain the hyperparameter and feature selection histories, respectively. These can be used to visualize the optimization results:

.. code-block:: python
	
	model.plot_hyper_opt()
	model.plot_feature_opt()

Example
-----------

We can load the diffuse_catalog and other_catalog files and create a Random Forest classifier as such:

.. code-block:: python
	
	import pandas
	import numpy as np
	from pyBIA import ensemble_model

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

	#Create 2D training data array 
	data_x = np.concatenate((blob, other))

	#Create 1D class label array
	labels_blob = np.array(['DIFFUSE']*len(blob))
	labels_other = np.array(['OTHER']*len(other))
	data_y = np.r_[labels_blob, labels_other]

	model = ensemble_model.classifier(data_x, data_y, clf='rf', impute=True, optimize=True)
	model.create()
	
Finally, we can make predictions using our optimized model:

.. code-block:: python

	prediction = model.predict(new_data)




