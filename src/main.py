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
from coarse_registration import fit_circle, iterative_icp

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
        self.scan = self.scan.loc[::10]
        
        if tilt:
            self.rotate_tilt()
            
        self.first_approximation(x, y)
        self.calculate_dem(2)
                    
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
        
        self.scan.loc[:, 'z_min'] = self.scan.groupby(['xbin',  'ybin']).z.transform(min)
        self.scan.loc[:, 'z_norm'] = self.scan.z - self.scan.z_min
        
        self.dem = self.scan.groupby(['xbin', 'ybin']).z.min().reset_index()

        return self
    
    def find_stems_2d(self, eps, min_samples, base_height=1.4, top_height=1.5,
                      max_residual=0.08, max_radius=0.2):
        z_slice = self.scan[self.scan.z_norm.between(base_height,
                                                     top_height)].copy()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(z_slice[['x', 'y']])
        z_slice.loc[:, 'labels'] = db.labels_
        
        stems = pd.DataFrame()

        for i, l in enumerate(np.unique(z_slice[(z_slice.labels != -1)].labels)):
            cluster = z_slice[(z_slice.labels == l)]

            try:
                xc, yc, R, residu = fit_circle(cluster.x.to_numpy(),
                                               cluster.y.to_numpy())            
            
                if (residu <= max_residual) & (R <= max_radius):
                    stems.loc[i, 'xc'] = xc
                    stems.loc[i, 'yc'] = yc
                    stems.loc[i, 'zc'] = np.mean(cluster.z)
                    stems.loc[i, 'R'] = R
                    stems.loc[i, 'residual'] = residu
                    stems.loc[i, 'labels'] = l
            except:
                pass
           
        return stems
    

if __name__ == "__main__":
    
    # Importing scan 1 and 2. As scan 2 (s2) is a tilt scan, set tilt to True
    # to perform the initial rotation to an upright position.
    s1 = Scan('../test_data/ScanPos037 - SINGLESCANS - 190710_205348.txt', 0, 0, False)
    s2 = Scan('../test_data/ScanPos038 - SINGLESCANS - 190710_205748.txt', 0, 0, True)
    
#    find stems in multiple height layers
    stems1 = s1.find_stems_2d(0.05, 200)
    stems2 = s2.find_stems_2d(0.05, 200)

    # Extracting the centre point coordinates from stems detected for scans 1
    # and 2.
    stem1_points = np.vstack((stems1.xc, stems1.yc, stems1.zc)).T
    stem2_points = np.vstack((stems2.xc, stems2.yc, stems2.zc)).T
    
    # Running Iterative Closest Point (ICP) to match stem points from scans.
    T, mean_error, total_matches = iterative_icp(stem2_points, stem1_points,
                                                 max_plane_rot=5)
    
    # Extracting point clouds from scans 1 and 2.
    scan1_points = np.vstack((s1.scan.x, s1.scan.y, s1.scan.z)).T
    scan2_points = np.vstack((s2.scan.x, s2.scan.y, s2.scan.z)).T
    
    # Using the transformation matrix (T) to transform 
    scan2_points_transformed = np.ones((scan2_points.shape[0], 4))
    scan2_points_transformed[:, 0:3] = np.copy(scan2_points)
    scan2_points_transformed = np.dot(T, scan2_points_transformed.T).T       