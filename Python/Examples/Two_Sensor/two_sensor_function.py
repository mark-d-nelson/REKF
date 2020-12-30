#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:09:41 2020

@author: megan_nelson
"""

import numpy as np

def sensor1transform(x,y):
    theta = np.arctan2(y, x)
    
    return theta

def sensor2transform(x,y):
    theta = np.arctan2(y, x-1.5)
    
    return theta

def sensor1gradient(x,y):
    u = np.zeros([2,1])
    u[0,0] = -y / (x*x + y*y)
    u[1,0] = x / (x*x + y*y)
    return u

def sensor2gradient(x,y):
    u = np.zeros([2,1])
    z = x-1.5
    u[0,0] = -y / (z*z + y*y)
    u[1,0] = z / (z*z + y*y)
    return u

