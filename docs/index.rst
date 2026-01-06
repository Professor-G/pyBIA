Welcome to pyBIA
================

.. admonition:: Documentation status (last updated |today|)
   :class: note

   This documentation is actively being developed and may change.

**pyBIA** (Blob Identification Algorithm) is an open-source Python framework for automated detection and classification of spatially extended, diffuse Lyman-alpha emission at high redshift (i.e., blob-like sources), such as Lyman-alpha blobs (LABs). The software was developed to support the analysis in **Godines, D. & Prescott, K. M. (2025, submitted)**. If you use pyBIA in a publication, we would appreciate citations to the paper as well as the `software release DOI <https://doi.org/10.5281/zenodo.17092327>`_.

By integrating source detection, aperture photometry, morphological segmentation, and machine learning, pyBIA provides an end-to-end pipeline for reducing large source catalogs into a prioritized candidate list for follow-up study. While optimized for high-redshift astronomy, its modular architecture makes it a flexible software tool for workflows requiring **image segmentation**, **anomaly detection**, or **classification**.

Reproducibility
---------------

Stochastic processes (e.g., model initialization, data shuffling) are controlled by a global seed attribute, ``SEED_NO`` (**1909** by default). You can override this during class initialization to enable reproducible runs (or set it to ``None`` for random runs). Note that while the classical machine-learning workflows are reproducible given a fixed seed, exact determinism for the deep-learning models will still vary unless deterministic TensorFlow settings are explicitly enabled.

Key Features
============

pyBIA is organized into four interconnected modules that streamline the transition from imaging to classified candidates.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Catalog Generation
      :link: https://pybia.readthedocs.io/en/latest/source/Catalog%20Generation.html
      :link-type: doc
      :text-align: center
      :class-card: intro-card

      .. image:: _static/catalog_img_link.png
         :alt: Catalog Generation
         :width: 150px
         :class: no-scaled-link

      Extract sources using segmentation maps, compute morphological moments, and generate photometric/morphological catalogs.

   .. grid-item-card:: Supervised Learning
      :link: https://pybia.readthedocs.io/en/latest/source/Supervised%20Learning%20Algorithms.html
      :link-type: doc
      :text-align: center
      :class-card: intro-card

      .. image:: _static/supervised_model_img_link.png
         :alt: Supervised Learning
         :width: 150px
         :class: no-scaled-link

      Train robust classifiers (XGBoost, Random Forest, etc.) with built-in **BorutaSHAP** feature selection and **Optuna** hyperparameter optimization.

   .. grid-item-card:: Anomaly Detection
      :link: https://pybia.readthedocs.io/en/latest/source/Anomaly%20Detection.html
      :link-type: doc
      :text-align: center
      :class-card: intro-card

      .. image:: _static/outlier_img_link.png
         :alt: Anomaly Detection
         :width: 150px
         :class: no-scaled-link

      Identify and remove imaging artifacts/outliers using Isolation Forests on extracted feature vectors (HOG, FFT, Wavelet).

   .. grid-item-card:: Deep Learning Classification
      :link: https://pybia.readthedocs.io/en/latest/source/Deep%20Learning%20Algorithms.html
      :link-type: doc
      :text-align: center
      :class-card: intro-card

      .. image:: _static/cnn_model_img_link.png
         :alt: Deep Learning Classification
         :width: 150px
         :class: no-scaled-link

      Train pre-built CNN architectures (AlexNet, ResNet18, VGG16) on single or multi-band imaging (up to 5 channels), with automated augmentation and cross-validation.

Quick Start
===========

Installation
------------

pyBIA requires **Python 3.12+**. Install the latest stable release via pip:

.. code-block:: bash

    pip install pyBIA

Alternatively, install the development version from GitHub:

.. code-block:: bash

    git clone https://github.com/Professor-G/pyBIA.git
    cd pyBIA
    pip install .

Citation
--------

If you use pyBIA in your research, please cite:

- **Godines, D. & Prescott, K. M. (2025, submitted)**
- **Godines (2025), Zenodo DOI: https://doi.org/10.5281/zenodo.17092327**

User Guide
==========

The pages below provide tutorials, API references, and high-level technical details on the program’s core functionality, as well as a dedicated section describing how **Godines, D. & Prescott, K. M. (2025, submitted)** was produced, including figure-by-figure generation details.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules
   :titlesonly:

   source/Catalog Generation
   source/Supervised Learning Algorithms
   source/Anomaly Detection
   source/Deep Learning Algorithms

.. toctree::
   :maxdepth: 2
   :caption: Case Studies
   :titlesonly:

   source/Godines & Prescott 2025

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :titlesonly:

   source/pyBIA
