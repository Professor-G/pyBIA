.. pyBIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA's documentation!
===============================

pyBIA is an open-source program for detecting diffuse Lyman-alpha emission in the high redshift universe, using a combination of machine learning ensemble and convolutional neural network algorithms. The program tools have been coded for general application, check out this `example <https://pybia.readthedocs.io/en/latest/source/Examples.html>`_ to learn how you can use pyBIA to create your own machine learning classifier. 

   
Installation
==================

The current stable version can be installed via pip:

.. code-block:: bash

    pip install pyBIA

You can also clone the development version:    

.. code-block:: bash

    git clone https://github.com/Professor-G/pyBIA.git
    python setup.py install
    pip install -r requirements.txt

Importing pyBIA 
==================

We have trained a Convolutional Neural Network using the high-level Keras API. Our model took ~3 days to train to a thousand epochs, and is included in the standard pyBIA installation. This classifier is called 'bw_model' as the DIFFUSE training sample includes diffuse objects in the blue broadband (see `Moire et al 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_). More models for different bands will be added in the future.

.. code-block:: python

    from pyBIA import cnn_model

    model = cnn_model.Classifier()
    model.load_bw_model()

With our model loaded, we can classify any 50x50 image using the predict function.

.. code-block:: python

    prediction = model.predict(data, normalize=True)

The output will either be 'DIFFUSE' or 'OTHER'. The input data can also be a 3-dimensional array containing multiple images.

How did we build pyBIA?
==================
Check out this `page <https://pybia.readthedocs.io/en/latest/source/Engineering%20pyBIA.html>`_ to learn more about the training data, as well as the machine learning architecture and latest performance. The latest performance 


Pages
==================
.. toctree::
   :maxdepth: 1

   source/Lyman-alpha Nebulae
   source/Engineering pyBIA
   source/Machine Learning
   source/Examples

Documentation
==================

Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/pyBIA