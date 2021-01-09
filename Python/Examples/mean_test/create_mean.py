#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 22:17:43 2021

@author: megan_nelson
"""

import numpy as np
import csv


state = np.asarray([-50, 200, 0, 1, 0, 0])


with open('mean_data.csv', 'w') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['Time', 'Range', 'Azimuth', 'Elevation', 'Range Variance', 'Azimuth Variance', 'Elevation Variance', 'X', 'Y', 'Z'])
    
    for cnt in range(100, 201):
        x = state[0] + float(cnt) * state[3]
        y = state[1] + float(cnt) * state[4]
        z = state[2] + float(cnt) * state[5]
        
        r = np.sqrt((x * x) + (y * y) + (z * z))
        theta = np.arctan(y / x)
        phi = np.arcsin(z / r)
        
        thewriter.writerow((cnt, np.random.normal(r, 0.004, 1)[0], \
                          np.random.normal(theta, 0.01, 1)[0], \
                          np.random.normal(phi, 0.01, 1)[0], \
                          0.004 * 0.004, 0.01 * 0.01, 0.01 * 0.01, x, y, z)) 
        
        