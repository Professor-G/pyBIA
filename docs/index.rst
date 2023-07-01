.. pyBIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA's documentation!
===============================

pyBIA is an open-source program for detecting diffuse Lyman-alpha emission in the high redshift universe, using a combination of machine learning tree-ensemble and convolutional neural network algorithms. Although developed as a tool for astronomers, the program has been coded for general application -- check out the `Examples <https://pybia.readthedocs.io/en/latest/source/Examples.html>`_ to learn how you can use pyBIA to create your own machine learning classifiers. 

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

Functionality
==================
The program provides three main functionalities:

-  Creating a catalog of astrophysical objects
-  Training a machine learning classifier with image moments
-  Training a machine learning classifier with single or multi-band imaging (up to 3 filters)

If you have a 2D array, but no positions, creating a catalog is quick and easy using the `catalog <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html>`_ module:

.. code-block:: python

    from pyBIA import catalog

    cat = catalog.Catalog(data)
    cat.create(save_file=True)

The X and Y pixel arguments can be input if source locations are known, with optional parameters available to control background subtraction, source detection thresholds, and flux calculations. If the error map is provided, the output catalog will contain the photometric error as well, which, after the catalog has been created, can be accessed via the ``cat`` class attribute which will be a dataframe containing all of the calculated flux and morphological features. These computed features can then be used to train a machine learning model using the `ensemble_model <https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/ensemble_model/index.html>`_ module. The only other accompanying method in the catalog class is `plot <https://pybia.readthedocs.io/en/latest/_modules/pyBIA/catalog.html#Catalog.plot>`_, which will output two subplots, the source and the corresponding segmentation object:

.. code-block:: python

    cat.plot()

.. figure:: _static/segm.png
    :align: center
    :class: with-shadow with-border
    :width: 600px
|

To learn about pyBIA's machine learning routines, refer to the Figures page which outlines how each Figure in the research paper was made, as well as the Examples page.


Pages
==================
.. toctree::
   :maxdepth: 1

   source/Lyman-alpha Nebulae
   source/Examples
   source/Figures 

Documentation
==================

Here is the documentation for all the modules:

.. toctree::
   :maxdepth: 1

   source/pyBIA