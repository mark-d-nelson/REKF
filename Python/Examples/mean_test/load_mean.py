#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:46:44 2021

@author: megan_nelson
"""


import sys 
sys.path.append('../..')

import Measurement as M
import csv
import mean_function 
import numpy as np

def GetMeasurements():
    measurements = []
    
    with open('mean_data.csv', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        cnt = 0
        for row in filereader:
            #print(', '.join(row))
            cnt += 1
            if cnt == 1:
                continue
            
            m = [float(row[1]), float(row[2]), float(row[3])]
            cov = np.zeros([3, 3])
            cov[0, 0] = float(row[4])
            cov[1, 1] = float(row[5])
            cov[2, 2] = float(row[6])
            truth = [float(row[7]), float(row[8]), float(row[9])]
            meas=M.Measurement(float(row[0]), m, cov, \
                               mean_function.Transformation, mean_function.Gradient, [1, 2], 'Sensor', truth)
            measurements.append(meas)
 
                
    return measurements