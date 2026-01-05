.. pyBIA documentation master file, created by
   sphinx-quickstart on Thu Mar 24 11:15:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyBIA
================

.. image:: _static/pyBIA_logo.png
   :alt: pyBIA Logo
   :align: center
   :width: 400px
   :class: no-scaled-link

.. raw:: html

   <br>

**pyBIA** (Blob Identification Algorithm) is an open-source Python framework designed for the automated detection and classification of spatially extended and diffuse Lyman-alpha emission at high redshift (i.e., blob-like objects), such as Lyman-alpha blobs (LABs).

By integrating source detection, aperture photometry, morphological segmentation, and state-of-the-art machine learning, pyBIA provides a complete pipeline for reducing millions of cataloged sources into a small candidate list for follow-up study. While optimized for high-redshift astronomy, its modular architecture makes it a powerful tool for any domain requiring **image segmentation**, **anomaly detection**, or **classification**.

Note that all stochastic processes (e.g., model initialization, data shuffling) are controlled by a global seed attribute, ``SEED_NO``, available in every core module. The default seed is set to **1909**. You can override this during class initialization to ensure your results are deterministic and reproducible (set to ``None`` for random runs.)

.. admonition:: Project Status
   :class: note

   This documentation is under active development (Last updated: January 2026).
   The software was developed to support the research in **Godines & Prescott (2025, submitted)**. If you use this code for publication, we would appreciate citations to the paper.

---

Key Features
============

pyBIA is built on four interconnected modules that streamline the transition from raw imaging to classified candidates.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Catalog Generation
      :link: source/Catalog%20Generation.html
      :text-align: center
      :class-card: intro-card

      .. image:: _static/catalog_img_link.png
         :alt: Catalog Generation
         :width: 150px
         :class: no-scaled-link

      Extract sources using segmentation maps, compute morphological moments, and generate photometric catalogs automatically.

   .. grid-item-card:: Supervised Learning
      :link: source/Supervised%20Learning%20Algorithms.html
      :text-align: center
      :class-card: intro-card

      .. image:: _static/supervised_model_img_link.png
         :alt: Supervised Learning
         :width: 150px
         :class: no-scaled-link

      Train robust classifiers (XGBoost, Random Forest, etc.) with built-in **BorutaSHAP** feature selection and **Optuna** hyperparameter optimization.

   .. grid-item-card:: Anomaly Detection
      :link: source/Anomaly%20Detection.html
      :text-align: center
      :class-card: intro-card

      .. image:: _static/outlier_img_link.png
         :alt: Anomaly Detection
         :width: 150px
         :class: no-scaled-link

      Identify and remove imaging artifacts or outliers using Isolation Forests on extracted feature vectors (HOG, FFT, Wavelet).

   .. grid-item-card:: Deep Learning Classification
      :link: source/Deep%20Learning%20Algorithms.html
      :text-align: center
      :class-card: intro-card

      .. image:: _static/cnn_model_img_link.png
         :alt: Deep Learning Classification
         :width: 150px
         :class: no-scaled-link

      Train pre-built convolutional neural networks (AlexNet, ResNet18, VGG16) on single or multi-band imaging (up to 3 channels), with automated data augmentation and cross-validation.

---

Quick Start
===========

Installation
------------

pyBIA requires **Python 3.12+**. Install the latest stable release via pip:

.. code-block:: bash

    pip install pyBIA

Alternatively, install the development version directly from GitHub:

.. code-block:: bash

    git clone https://github.com/Professor-G/pyBIA.git
    cd pyBIA
    pip install .

Citation
--------

If you use pyBIA in your research, please cite the following paper:

    **Godines, D. & Prescott, K. M. (2025, submitted).** 

---

User Guide
==========

These sections provide detailed tutorials, API references, and the specific workflows used in our research.
To learn more about pyBIA's machine learning routines, consult the pages listed below. These pages provide high-level technical details on the program’s core functionality, as well as a dedicated section describing how **Godines, D. & Prescott, K. M. (2025, submitted)**  was produced, including figure-by-figure generation details.

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