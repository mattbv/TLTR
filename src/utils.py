#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:13 2019

@author: matheus
"""

import numpy as np

def pairwise_angle(p1, p2):
    return np.arctan2(p2[0] - p1[0], p2[1] - p1[1])


def point_features(cloud):
    
   nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree',
                           n_jobs=-1).fit(cloud)
   dist, idx = nbrs.kneighbors(cloud, return_distance=True)
   
   angles = np.zeros([idx.shape[0], 2])
   for i in idx:
       angles[i[0], 0] = pairwise_angle(cloud[i[0]], cloud[i[1]])
       angles[i[0], 1] = pairwise_angle(cloud[i[0]], cloud[i[2]])
        
   return dist[:, 1:], angles