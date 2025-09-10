# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:30:11 2022

@author: danielgodinez
"""
from setuptools import setup, find_packages

setup(
    name="pyBIA",
    version="1.51",
    author="Daniel Godines",
    author_email="danielgodinez123@gmail.com",
    description="Machine learning-based framework for Lyalpha Blob Detection",
    long_description="A machine learning pipeline for detecting spatially extended, diffuse emission in multi-band broadband imaging.",
    license='GPL-3.0',
    url = "https://github.com/Professor-G/pyBIA",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages('.'),
    install_requires=[
        "numpy==2.1.3",
        "pandas==2.3.1",
        "matplotlib==3.10.3",
        "astropy==7.1.0",
        "photutils==2.2.0",
        "scipy==1.16.0",
        "tensorflow==2.19.0",
        "progress==1.6.1",
        "joblib==1.5.1",
        "scikit-learn==1.7.1",
        "xgboost==3.0.2",
        "optuna==4.4.0",
        "optuna-integration==4.4.0",
        "shap==0.48.0",
        "tqdm==4.67.1",
        "scikit-image==0.25.2",
        "PyWavelets==1.8.0",
    ],
    python_requires='>=3.12',
    include_package_data=True,
    test_suite="nose.collector",
)
