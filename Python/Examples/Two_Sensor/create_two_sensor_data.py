#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:55:38 2020

@author: megan_nelson
"""

import numpy as np
import csv
piovertwo = np.pi / 2
pioverfour = np.pi / 4
R = 0.0001

with open('two_sensor_data.csv', 'w') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['Sensor', 'Variance', 'Measurement'])
    
    for cnt in range(0, 10):
      
      thewriter.writerow(('S1', R, np.random.normal(pioverfour, np.sqrt(R), 1)[0]))  
    
      thewriter.writerow(('S2', R, np.random.normal(piovertwo, np.sqrt(R), 1)[0]))  
