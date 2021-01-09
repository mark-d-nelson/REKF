#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:55:39 2021

@author: megan_nelson
"""

import numpy as np

def TemporalDerivative(t, x):
    A = np.zeros([6,6])
    A[0:3, 3:6] = np.eye(3)
    
    y = np.matmul(A, x)
    return y


def Transformation(s):
    Y = np.zeros(3)
    Y[0] = np.sqrt((s[0] * s[0]) + (s[1] * s[1]) + (s[2] * s[2]))
    Y[1] = np.arctan2(s[1], s[0])
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
    
    H[1, 0] =  -s[1] / p
    H[1, 1] = s[0] / p
    
    H[2, 0] = -(s[0] * s[2]) / (r * P)
    H[2, 1] = -(s[1] * s[2]) / (r * P)
    H[2, 2] = -(s[2] * s[2]) / (r * P) + 1. / P
    return H
