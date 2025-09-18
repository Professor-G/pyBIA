.. pyBIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA's documentation!
===============================

.. admonition:: Under Construction (last updated 2025-09-17)

   This documentation is still being written and may change frequently!

pyBIA is an open-source program for detecting diffuse Lyman-alpha emission in the high redshift universe, using a combination of machine learning tree-ensemble and convolutional neural network algorithms. Although developed as a tool for astronomers, the program has been coded for general application. This documentation explains how you can use pyBIA to build and train your own machine-learning classifiers.


Functionality
==================
The program provides four core modules:

.. grid:: 2
   :gutter: 2

   .. grid-item-card::
      :text-align: center

      **Catalog Generation**

      .. image:: _static/catalog_img_link.png
         :alt: Catalog Generation
         :target: https://pybia.readthedocs.io/en/latest/source/Catalog%20Generation.html
         :width: 200px

      Build a morphological catalog from imaging data using image segmentation.

   .. grid-item-card::
      :text-align: center

      **Supervised Learning**

      .. image:: _static/supervised_model_img_link.png
         :alt: Supervised Learning
         :target: https://pybia.readthedocs.io/en/latest/source/Supervised%20Learning%20Algorithms.html
         :width: 200px

      Train and optimize supervised machine learning models for classification.

   .. grid-item-card::
      :text-align: center

      **Anomaly Detection**

      .. image:: _static/outlier_img_link.png
         :alt: Anomaly Detection
         :target: https://pybia.readthedocs.io/en/latest/source/Anomaly%20Detection.html
         :width: 200px

      Identify unusual sources with an unsupervised outlier-removal pipeline.

   .. grid-item-card::
      :text-align: center

      **Deep Learning Classification**

      .. image:: _static/cnn_model_img_link.png
         :alt: Deep Learning Classification
         :target: https://pybia.readthedocs.io/en/latest/source/Deep%20Learning%20Algorithms.html
         :width: 200px

      Convolutional neural networks for single or multi-band imaging (up to 3 channels).


Installation
==================

The current stable version can be installed via pip:

.. code-block:: bash

    pip install pyBIA

You can also clone the development version:    

.. code-block:: bash

    git clone https://github.com/Professor-G/pyBIA.git
    cd pyBIA
    pip install .

**NOTE:** The program requires Python3.12+.

To learn more about pyBIA's machine learning routines, consult the pages listed below. These pages provide high-level technical details on the program’s core functionality, as well as a dedicated section describing how Godines et al. (2025) was produced, including figure-by-figure generation details.

Pages
==================
.. toctree::
   :maxdepth: 1

   source/Catalog Generation
   source/Supervised Learning Algorithms
   source/Anomaly Detection
   source/Deep Learning Algorithms
   source/Godines et al 2025

Documentation
==================

Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/pyBIA
