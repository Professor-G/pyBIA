#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 22:59:04 2021

@author: daniel
"""

import numpy as np
import random
import os

from data_processing import process_class
from data_augmentation import augmentation, resize
from pyBIA import pyBIA_model, predict

def all_chars(x):
    return(x[:])


path= '/Users/daniel/Desktop/NDWFS_Tiles/IMAGE_DATA/BLOBS/Bw'

filenames = os.listdir(path)
filenames = sorted(filenames, key=all_chars)


diffuse_data=[]
group_data=[]
candidates_85_group = []
candidates_85_diffuse = []


for f in filenames:
    if f[0:4] == 'DIFF':
        filepath = os.path.join(path, f)
        data = np.loadtxt(filepath)
        if f[-8:] == '_85_blob':
            candidates_85_diffuse.append(data)
        else:
            diffuse_data.append(data)
    elif f[:4] == 'GROU':
        filepath = os.path.join(path, f)
        data = np.loadtxt(filepath)
        if f[-8:] == '_85_blob':
            candidates_85_group.append(data)
        else:
            group_data.append(data)
            
diffuse_data = np.array(diffuse_data)
group_data = np.array(group_data)
candidates_85_group = np.array(candidates_85_group)
candidates_85_diffuse = np.array(candidates_85_diffuse)

other_data=[]
path= '/Users/daniel/Desktop/NDWFS_Tiles/IMAGE_DATA/OTHERS/Bw'

filenames = os.listdir(path)
filenames = sorted(filenames, key=all_chars)

bad_files=[]
for f in filenames:
 print(f)
 if f[:4] == 'OTHE':
    filepath = os.path.join(path, f)
    try:
        data = np.loadtxt(filepath)
        other_data.append(data)
    except:
        print(filepath)
        bad_files.append(f)
        pass
        
other = np.array(other_data[:781])
blob = np.r_[diffuse_data, group_data]
