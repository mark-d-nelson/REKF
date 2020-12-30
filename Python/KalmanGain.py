# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:44:16 2020

@author: MDnelson
"""

import numpy as np
from scipy.linalg import cholesky

def AngularDomain( x, angular_index ):
        
    if not angular_index:
        return x
    
    for i in angular_index:
        x[i] = (x[i] + np.pi)%(2. * np.pi) - np.pi
        
    return x

def IndentityFunction(a,b,x):
    return x.copy()






class KalmanGain:
    
    def __init__( self, temporal_function, state_dim, process_noise, angular_index = [] ):
    
        self.PrevTime          = 0.
        self.NextTime          = 0.
        if not temporal_function:
            self.TemporalFunction  = temporal_function
        else:
            self.TemporalFunction  = IndentityFunction
        
        if not angular_index:
            if np.isscalar( angular_index ):
                angular_index = [angular_index]
            angular_index = np.asarray(angular_index)
        self.AngularIndex      = angular_index
        
        self.MeasurementObject = []
        
        self.dimension       = state_dim
        
        if not process_noise:
            if np.isscalar( process_noise ):
                process_noise = [[process_noise]]
            process_noise = np.asarray(process_noise)
        self.process_noise   = process_noise
        
        self.extend_dim      = self.dimension + len(self.process_noise)
        self.U               = np.zeros([self.extend_dim, self.extend_dim], dtype='float')
        
        self.sigma_length    = (1e-3)**2 * self.extend_dim
        
        # constant of .25 = .5 for the two particles +/- U
        # and a second .5 for a transpose to ensure symmetry
        self.cov_weight      = 0.25 / self.sigma_length
        
    
    def decomposeCovariance(self, cov):
        if len(self.process_noise) > 0:
            P = np.zeros([self.extend_dim, self.extend_dim], dtype='float')
            P[:self.dimension, :self.dimension] = cov
            P[self.dimension:,self.dimension:] = self.process_noise
            self.U = cholesky(self.sigma_length * P)
        else:
            self.U = cholesky(self.sigma_length * cov)
        
    # Input variable x is from current measurement
    def GetChangeInState( self, x ):
        
        cov = self.GetCovariance( x )
        H   = self.MeasurementObject.GradientFunction( x )
        Q   = np.matmul(cov, np.transpose(H))
        u   = np.linalg.solve(np.matmul(H, Q) + self.MeasurementObject.Covariance, self.innovation)
        dx  = np.dot(Q, u)
        
        dx = AngularDomain( dx, self.AngularIndex )
        
        return dx
    
    # Input time, x, and covariance are from previous measurement
    # Input meas_obj is current measurement
    def SetGainParam(self, time, x, cov, meas_obj):
        
        self.decomposeCovariance(cov)
        self.MeasurementObject = meas_obj
        self.PrevTime   = time
        self.NextTime   = self.MeasurementObject.Time
        self.innovation = self.MeasurementObject.Estimate - \
            self.MeasurementObject.TransformFunction( self.GetState(x) )
            
        self.innovation = AngularDomain( self.innovation, self.MeasurementObject.AngularIndex)
        
    # Input variable x is from current measurement
    def UpdateCovariance(self, x):
        
        cov = self.GetCovariance( x )
        H   = self.MeasurementObject.GradientFunction( x )
        Q   = np.matmul(cov, np.transpose(H))
        D   = np.linalg.inv(np.matmul(H, Q) + self.MeasurementObject.Covariance)
        C   = np.eye(len(cov)) - np.matmul(Q, D)
        cov = np.matmul(C, cov)
        cov = 0.5 * (cov + np.transpose(cov))
        
        return cov
        
    # Input variable x is from current measurement
    def GetCovariance(self, x):
        
        y = np.zeros(self.extend_dim, dtype='float')
        y[:self.dimension] = self.TemporalFunction(self.NextTime, self.PrevTime, x)
        
        cov = np.zeros([self.dimension, self.dimension], dtype='float')
        for k in range(0,self.extend_dim):
            residual = self.TemporalFunction(self.PrevTime, self.NextTime, \
                                             y + self.U[k])[:self.dimension] - x
            residual = AngularDomain( residual, self.AngularIndex )
            cov += np.outer(residual, residual)
            
            residual = self.TemporalFunction(self.PrevTime, self.NextTime, 
                                             y - self.U[k])[:self.dimension] - x
            residual = AngularDomain( residual, self.AngularIndex )
            cov += np.outer(residual, residual)
        cov = self.cov_weight * (cov + np.transpose(cov))
        
        return cov
    
    # Input variable x is from previous measurement
    def GetState( self, x ):
        y = self.TemporalFunction(self.PrevTime, self.NextTime, x)
        
        y = AngularDomain( y, self.AngularIndex)
        
        return y