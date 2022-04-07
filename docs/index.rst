.. MicroLIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA's documentation!
===============================

pyBIA is an open-source program for Lyman-alpha blob detection in wide-field surveys. This engine uses a machine learning convolutional neural network model, trained with 50x50 images; therefore when using the standard pyBIA models, the data input for classification must also be 50x50 pixels.

   
Installation
==================

The current stable version can be installed via pip:

.. code-block:: bash

    pip install pyBIA


How did we build pyBIA?
==================
Check out this page to learn how more about the motivation behind pyBIA, how it was constructed and trained,
and how you can use pyBIA modules to replicate our work and create your own Convolutional Neural Networks or add
data of newly discovered Lyman-alpha nebulae to our existing models!

Importing pyBIA 
==================

We have trained a Convolutional Neural Network using the high-level Keras API. Our model took ~8 days to train and is included in the standard pyBIA installation. This classifier is called 'bw_model' as the DIFFUSE training data sample includes diffuse objects in the NDWFS Bw band footprint (see `Moire et al. 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_). We hope to add more models for different bands in the future.

.. code-block:: python

    from pyBIA import models

    model = models.bw_model()

With our model loaded, we can classify any 50x50 2-dimensional matrix using the predict function.

.. code-block:: python

    prediction = models.predict(data, model)

The output will either be 'DIFFUSE' or 'OTHER'. The data can be 3-dimensional as well, in which
case it would contain multiple 50x50 matrices.


Documentation
==================

Here is the documentation for all the modules, see :any:`source/_Engineering_pyBIA` for the details

.. toctree::
   :maxdepth: 1

   source/pyBIA


Pages
==================
.. toctree::
   :maxdepth: 2

   source/Installation
   source/Engineering pyBIA
   source/Examples
