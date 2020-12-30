#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:57:24 2020

@author: megan_nelson
"""

import Measurement as M
import csv
import two_sensor_function as tsf

def GetMeasurements():
    measurements = []
    
    with open('two_sensor_data.csv', newline='') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        cnt = 0
        for row in filereader:
            #print(', '.join(row))
            cnt += 1
            if cnt == 1:
                continue
            
            if row[1] == 'S1':
                
                meas=M.Measurement(float(row[0]), float(row[3]), float(row[2]), \
                                   tsf.sensor1transform, tsf.sensor1gradient, 0, row[1])
            else:
                meas=M.Measurement(float(row[0]), float(row[3]), float(row[2]), \
                                   tsf.sensor2transform, tsf.sensor2gradient, 0, row[1])
            
            measurements.append(meas)
 
                
    return measurements

