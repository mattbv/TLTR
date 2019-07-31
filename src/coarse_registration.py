#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:03:13 2019

@author: matheus
"""
import itertools as it
import numpy as np
from icp import icp
from ransac_circle import RANSAC

def iterative_icp(key_points, base_points, min_matches=3):
    
    max_n_samples = np.min([key_points.shape[0], base_points.shape[0]])
    sample_numbers = np.arange(min_matches, max_n_samples + 1)
        
    key_ids = np.arange(key_points.shape[0], dtype=int)
    base_ids = np.arange(base_points.shape[0], dtype=int)
    
    min_error = np.inf
    optimum_T = np.eye(4)
    total_matches = np.inf
    for sn in sample_numbers[::-1]:
        for key_sids in it.combinations(key_ids, sn):
            key_sids = np.array(key_sids)
            for base_sids in it.combinations(base_ids, sn):
                base_sids = np.array(base_sids)
                T, error, i = icp(key_points[key_sids], base_points[base_sids])
                mean_error = np.mean(error)
                if mean_error < min_error:
                    min_error = mean_error
                    optimum_T = T
                    total_matches = sn
                    
    return optimum_T, min_error, total_matches

#ADD FILTER SO IT CANT FLIP UPSIDE
    
def fit_circle(x, y, n_iter=100):
    
    ransac = RANSAC(x, y, n_iter)
    ransac.execute_ransac()
    xc, yc, R = ransac.best_model[0], ransac.best_model[1], ransac.best_model[2]
    
    Ri = np.sqrt(((x - xc) ** 2) + ((y - yc) ** 2))
    residu = np.sum((Ri - R) ** 2)
    
    return xc, yc, R, residu