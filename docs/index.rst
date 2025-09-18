.. pyBIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA's documentation!
===============================

.. admonition:: Under Construction (last updated 2025-09-17)

   This documentation is still being written and may change frequently!

pyBIA (Blob Identification Algorithm) is an open-source framework for machine learning detection of spatially extended and diffuse Lyman-alpha emission at high redshift (i.e., blob-like objects). Combining a variety of machine learning methods, pyBIA can reduce millions of cataloged sources to a small candidate list for follow-up study. 

Although designed for astronomers, its modular architecture make it adaptable to any domain where large-scale image segmentation and/or classification are required. This documentation walks you through installation, data preparation, model training, and optimization so you can quickly build your own machine-learning classifiers with pyBIA.

This software was created to conduct the research presented in Godines & Prescott (2025). If you use this code for publication, we would appreciate citations to the paper.

Functionality
==================
The program provides four core modules:

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Catalog Generation
      :link: https://pybia.readthedocs.io/en/latest/source/Catalog%20Generation.html
      :text-align: center

      .. image:: _static/catalog_img_link.png
         :alt: Catalog Generation
         :width: 200px

      Build a morphological catalog from imaging data using image segmentation.

   .. grid-item-card:: Supervised Learning
      :link: https://pybia.readthedocs.io/en/latest/source/Supervised%20Learning%20Algorithms.html
      :text-align: center

      .. image:: _static/supervised_model_img_link.png
         :alt: Supervised Learning
         :width: 200px

      Train and optimize supervised machine learning models for classification.

   .. grid-item-card:: Anomaly Detection
      :link: https://pybia.readthedocs.io/en/latest/source/Anomaly%20Detection.html
      :text-align: center

      .. image:: _static/outlier_img_link.png
         :alt: Anomaly Detection
         :width: 200px

      Identify unusual sources with an unsupervised outlier-removal pipeline.

   .. grid-item-card:: Deep Learning Classification
      :link: https://pybia.readthedocs.io/en/latest/source/Deep%20Learning%20Algorithms.html
      :text-align: center

      .. image:: _static/cnn_model_img_link.png
         :alt: Deep Learning Classification
         :width: 200px

      Train convolutional neural networks on single or multi-band imaging (up to 3 channels).


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

To learn more about pyBIA's machine learning routines, consult the pages listed below. These pages provide high-level technical details on the program’s core functionality, as well as a dedicated section describing how Godines & Prescott (2025) was produced, including figure-by-figure generation details.

Pages
==================
.. toctree::
   :maxdepth: 1

   source/Catalog Generation
   source/Supervised Learning Algorithms
   source/Anomaly Detection
   source/Deep Learning Algorithms
   source/Godines & Prescott 2025

Documentation
==================

Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/pyBIA
