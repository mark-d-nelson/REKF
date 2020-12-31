#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:01:04 2020

@author: megan_nelson
"""

import Measurement as M
import csv
import ruf_tracking_function as rufunction
import numpy as np

def GetMeasurements():
    measurements = []
    
    with open('ruf_tracking_data.csv', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        cnt = 0
        for row in filereader:
            #print(', '.join(row))
            cnt += 1
            if cnt == 1:
                continue
            
            m = [float(row[2]), float(row[4]), float(row[6])]
            cov = np.zeros([3, 3])
            cov[0, 0] = float(row[3])
            cov[1, 1] = float(row[5])
            cov[2, 2] = float(row[7])
            meas=M.Measurement(float(row[0]), m, cov, \
                               rufunction.Transformation, rufunction.Gradient, [1, 2], row[1])
            measurements.append(meas)
 
                
    return measurements
