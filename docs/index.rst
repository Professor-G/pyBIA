.. pyBIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA's documentation!
===============================

Lyman-alpha nebulae are extremely rare, to aid in the search for these objects we have created pyBIA, a deep learning classifier for detecting Lyman-alpha blobs using single band imaging. The latest assesment output a false-positive rate of ~4%.

The program tools have been coded for general application, check out this `example <https://pybia.readthedocs.io/en/latest/source/Examples.html>`_ to learn how you can use pyBIA modules to create your own machine learning image classifier. 

   
Installation
==================

The current stable version can be installed via pip:

.. code-block:: bash

    pip install pyBIA

You can also clone the development version:    

.. code-block:: bash

    git clone https://github.com/Professor-G/pyBIA.git
    python setup.py -U install
    pip install -r requirements.txt


How did we build pyBIA?
==================
Check out this `page <https://pybia.readthedocs.io/en/latest/source/Engineering%20pyBIA.html>`_ to learn more about the training data, as well as the machine learning architecture and latest performance. 


Importing pyBIA 
==================

We have trained a Convolutional Neural Network using the high-level Keras API. Our model took ~8 days to train to a thousand epochs, and is included in the standard pyBIA installation. This classifier is called 'bw_model' as the DIFFUSE training data sample includes diffuse objects in the blue broadband (see `Moire et al 2012 <https://arxiv.org/pdf/1111.2603.pdf>`_). We hope to add more models for different bands in the future.

.. code-block:: python

    from pyBIA import cnn_model

    model = cnn_model.bw_model()

With our model loaded, we can classify any 50x50 image using the predict function.

.. code-block:: python

    prediction = models.predict(data, model, normalize=True, min_pixel=1000, max_pixel=1600)

The output will either be 'DIFFUSE' or 'OTHER'. The input data can also be a 3-dimensional array containing multiple images.

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