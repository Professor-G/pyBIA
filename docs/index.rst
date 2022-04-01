.. LIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA's documentation!
===============================

pyBIA is an open-source program for Lyman-alpha blob detection in wide-field surveys. This engine uses the machine learning convolutional neural network model, trained with 50x50 images.

Installation
==================

The current stable version can be installed via pip:

.. code-block:: bash

    pip install pyBIA


Importing pyBIA 
==================
In this example we will load the standard pyBIA model for blue broadband images, and use this model to classify low redshift lyman-alpha blobs, known as Green Bean galaxies. 

.. code-block:: python

    from pyBIA import pyBIA

    model = pyBIA.bw_model()

This is our CNN model, which we can now use to classify any 50x50 image.

.. code-block:: python

    prediction = pyBIA.predict(data, model)

The data format must be a 2-dimensional array.

Pages
----------------------
.. toctree::
   :maxdepth: 2

   source/Installation
   source/Conventions
   source/Examples

modules details
----------------------

Here is the (hopefully up-to-date) documentation
for all submodules.

.. toctree::
   :maxdepth: 1

   source/pyBIA

