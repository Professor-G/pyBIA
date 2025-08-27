[![Documentation Status](https://readthedocs.org/projects/pybia/badge/?version=latest)](https://pybia.readthedocs.io/en/latest/?badge=latest)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/LGPL-3.0)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)

# pyBIA

A machine learning classification pipeline for detecting Lyman-alpha blobs in wide-field surveys using multi-band broadband data.

The pyBIA framework consists of four main modules

* [catalog](https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/catalog/index.html) : Used to generate a catalog of morphological and intensity-based characteristics using image segmentation.
* [ensemble_model](https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/ensemble_model/index.html) : To train and optimize (including feature selection and hyperparameter tuning) a supervised learning classifier.
* [outlier_detection](https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/outlier_detection/index.html) : Used to extract features for image-based anomaly detection, and training an unsupervised anomaly detection algorithm.
* [cnn_model](https://pybia.readthedocs.io/en/latest/autoapi/pyBIA/cnn_model/index.html) : For processing multi-band imaging data (including pre-processing and data augmentation) and training a deep learning image classifier.

For more information including examples of how to use the code, please see the [documentation](https://pybia.readthedocs.io/en/latest/) page.

# Installation

The latest stable version can be installed via pip.

```
    $ pip install pyBIA
```

# [Documentation](https://pybia.readthedocs.io/en/latest/)

For technical details and an example of how to implement pyBIA, including how it was used in Godines & Prescott (2025), check out our [Documentation](https://pybia.readthedocs.io/en/latest/).


# How to Contribute?

Want to contribute? Bug detections? Comments? Suggestions? Please email us : danielgodinez123@gmail.com, mkpresco@nmsu.edu
