# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:30:11 2022

@author: danielgodinez
"""
from setuptools import setup, find_packages, Extension


setup(
    name="pyBIA",
    version="1.1.5",
    author="Daniel Godines",
    author_email="danielgodinez123@gmail.com",
    description="Convolutional Neural Network for Ly-alpha Blob Detection",
    license='GPL-3.0',
    url = "https://github.com/Professor-G/pyBIA",
    classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3',	   
],
    packages=find_packages('.'),
    install_requires = ['numpy','tensorflow','scikit-learn','scikit-optimize','scipy','photutils', 
        'matplotlib', 'progress', 'pandas','missingpy', 'optuna', 'boruta', 'astropy', 'xgboost', 'BorutaShap',
        'scikit-plot', 'dill'],
    python_requires='>=3.7,<4',
    include_package_data=True,
    test_suite="nose.collector",
    package_data={
    '': ['Bw_CNN_Model.h5'],
},

)
