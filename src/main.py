#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:41:43 2019

@author: matheus
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

class Scan:
    
    def __init__(self, path, x, y, tilt=False):
  
        self.scan = pd.read_csv(path, dtype=float)
        self.scan.rename(columns={'Target Count[]':'cnt',
                                  'Scan Segment[]':'scan_segment',
                                  'XYZ[0][m]':'x',
                                  'XYZ[1][m]':'y',
                                  'XYZ[2][m]':'z',
                                  'Timestamp[s]':'timestamp',
                                  'Amplitude[dB]':'amp',
                                  'Deviation[]':'dev',
                                  'Reflectance[dB]':'refl',
                                  'Target Index[]':'tgt_idx'}, inplace=True)
    
        self.scan = self.scan[(self.scan.dev <= 10) & (self.scan.tgt_idx == 1)]
        self.filter_outliers(0.05)
        self.scan = self.scan.loc[::100]
        
        if tilt:
            self.rotate_tilt()
            
        self.first_approximation(x, y)
        
            
#        self.coarse()
        
    def rotate_tilt(self):
        
        self.scan.loc[:, 'a'] = 1
        R = np.array([[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        self.scan[['x', 'y', 'z', 'a']] = np.dot(R, self.scan[['x', 'y', 'z', 'a']].T).T
        
        return self
        
    def first_approximation(self, x, y):
        
        self.scan.x = self.scan.x.subtract(self.scan.x.mean()) + x
        self.scan.y = self.scan.y.subtract(self.scan.y.mean()) + y
        
        return self 
    
    def filter_outliers(self, dist_threshold):
        
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree',
                                n_jobs=-1).fit(self.scan[['x', 'y', 'z']])
        dist, idx = nbrs.kneighbors(self.scan[['x', 'y', 'z']])
        
        self.scan = self.scan[(np.mean(dist, axis=1) <= dist_threshold)]
        
        return self

    def calculate_dem(self, bin_size):
        
        self.scan.loc[:, 'xbin'] = np.floor(self.scan.x / bin_size).astype(int)
        self.scan.loc[:, 'ybin'] = np.floor(self.scan.y / bin_size).astype(int)
        
        self.scan.loc[:, 'z_norm'] = self.scan.groupby(['xbin', 'ybin']).z.transform(min)
        
        self.dem = self.scan.groupby(['xbin', 'ybin']).z.min().reset_index()

        
        return self
    
    def find_stems_2d(self, eps, min_samples):
        canopy = self.scan[self.scan.z_norm.between(1.2, 1.5)].copy()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(canopy[['x', 'y']])
        canopy.loc[:, 'labels'] = db.labels_
        
        return canopy
        