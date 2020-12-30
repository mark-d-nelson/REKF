#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:09:41 2020

@author: megan_nelson
"""

import numpy as np

def sensor1transform(x):
    theta = np.arctan2(x[1], x[0])
    
    return theta

def sensor2transform(x):
    theta = np.arctan2(x[1], x[0]-1.5)
    
    return theta

def sensor1gradient(x):
    u = np.zeros([1,2])
    r = x[0]*x[0] + x[1]*x[1]
    u[0,0] = -x[1] / r
    u[0,1] =  x[0] / r
    return u

def sensor2gradient(x):
    u = np.zeros([1,2])
    z = x[0]-1.5
    r = z*z + x[1]*x[1]
    u[0,0] = -x[1] / r
    u[0,1] =  z    / r
    return u

