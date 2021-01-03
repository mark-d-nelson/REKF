#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:14:01 2020

@author: megan_nelson
"""
import sys 
sys.path.append('../..')

import ButcherTableau as BT
import numpy as np
import csv
import ruf_tracking_function as rtf

#All Rs are variance
R_range = 0.1 * 0.1
R_azimuth = (np.pi * 0.1 / 180.) ** 2
R_elevation = (np.pi * 0.1 / 180.) ** 2
R = np.pi * np.pi * 1e-5

pbt = BT.ButcherTableauExplicitMethods('RK4')

with open('ruf_tracking_data.csv', 'w') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['Time', 'Sensor', 'Range', 'Range Variance', 'Azimuth', \
                        'Azimuth Variance', 'Elevation', 'Elevation Variance', 'X', 'Y', 'Z'])
    time       = 0.
    state_est  = np.asarray( [100., 0., 5., 0., 0., -.05 ] )
    
    for cnt in range(0, 1001):
        meas = rtf.Transformation(state_est)
      
        # thewriter.writerow((time, 'S1', meas[0], \
        #                   R_range, meas[1], \
        #                   R_azimuth, meas[2], \
        #                   R_elevation, state_est[0], state_est[1], state_est[2]))
            
        thewriter.writerow((time, 'S1', np.random.normal(meas[0], np.sqrt(R_range), 1)[0], \
                          R_range, np.random.normal(meas[1], np.sqrt(R_azimuth), 1)[0], \
                          R_azimuth, np.random.normal(meas[2], np.sqrt(R_elevation), 1)[0], \
                          R_elevation, state_est[0], state_est[1], state_est[2]))  
        time +=  1
        state_est = pbt.IntegrationTimeIndependent(state_est, rtf.TemporalDerivative2)
        #state_est = rtf.Temporal(state_est)
        
        