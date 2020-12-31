#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:25:27 2020

@author: megan_nelson
"""

import numpy as np

def Temporal(x):
    T = np.asarray( \
   [[1.0, 0.0, 1.33099992e-09, 9.99999193e-01, 0.0, 1.09999989e-03], \
    [0.0, 9.99999395e-01, 0.0, 0.0, 9.99999798e-01, 0.0], \
    [0.0, 0.0, 1.00000181e+00, -1.09999989e-03, 0.0, 9.99999798e-01], \
    [0.0, 0.0, 3.99299960e-09, 9.99997580e-01, 0.0, 2.19999956e-03], \
    [0.0, -1.20999976e-06, 0.0, 0.0, 9.99999395e-01, 0.0], \
    [0.0, 0.0, 3.62999927e-06, -2.19999956e-03, 0.0, 9.99999395e-01] \
    ])
    
    y = np.matmul(T, x)
    return y

def Transformation(s):
    Y = np.zeros(3)
    Y[0] = np.sqrt((s[0] * s[0]) + (s[1] * s[1]) + (s[2] * s[2]))
    Y[1] = np.arctan2(s[0], s[1])
    Y[2] = np.arcsin(s[2] / Y[0])
    return Y

def Gradient(s):
    H = np.zeros([3, 6])
    p = (s[1] * s[1]) + (s[0] * s[0])
    r = p + (s[2] * s[2])
    R = np.sqrt(r)
    P = np.sqrt(p)
    
    H[0, 0] = s[0] / R
    H[0, 1] = s[1] / R
    H[0, 2] = s[2] / R
    
    H[1, 0] =  s[1] / p
    H[1, 1] = -s[0] / p
    
    H[2, 0] = -(s[0] * s[2]) / (r * P)
    H[2, 1] = -(s[1] * s[2]) / (r * P)
    H[2, 2] = -(s[2] * s[2]) / (r * P) + 1. / P
    return H

