#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 09:06:08 2020

@author: Mark
"""

import numpy as np

class Measurement:
    def __init__( self, time, meas, cov, transform_func, gradient_func, 
                 angular_index = [], label='default sensor', truth = [] ):
        
        if np.isscalar( meas ):
            meas = [meas]
        if np.isscalar( cov ):
            cov = [[cov]]
        
        meas = np.asarray( meas )
        cov  = np.asarray( cov  )
            
        if not angular_index:
            if np.isscalar( angular_index ):
                angular_index = [angular_index]
            angular_index = np.asarray(angular_index)
            
        if not truth:
            if np.isscalar( truth ):
                truth = [truth]
            truth = np.asarray(truth)
        
        self.Label             = label
        self.Time              = time
        self.Estimate          = meas
        self.Covariance        = cov
        self.TransformFunction = transform_func
        self.GradientFunction  = gradient_func
        self.AngularIndex      = angular_index
        self.Truth             = truth
        