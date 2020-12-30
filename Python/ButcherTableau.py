# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:32:07 2020

@author: MDnelson
"""

import numpy as np

# Explicit methods only -> diagonal and upper of A is zero, and so is first value in c
class ButcherTableauExplicitMethods:
    
    def __init__(self, method = 'default', alpha = 0.5 ):
        
        # Alpha is used only for Generic methods and is not a required parameters
        
        self.tableaus = {
            'Forward Euler'    : self.ForwardEuler,
            'Midpoint'         : self.Midpoint,
            'Heun'             : self.Heun2,
            'Ralston'          : self.Ralston2,
            'Generic 2nd Order': self.Generic2,  # alpha value defaulted to .5
            'Kutta 3rd Order'  : self.Kutta3,
            'Generic 3rd Order': self.Generic3,  # alpha value defaulted to .5
            'Heun 3rd Order'   : self.Heun3,
            'Ralston 3rd Order': self.Ralston3,
            'SSPRK3'           : self.SSPRK3,
            'RK4'              : self.RK4,
            'Ralston 4th Order': self.Ralston4,
            'Kutta 4th Order'  : self.Kutta4,
            'Heun-Euler'       : self.HeunEuler,       # 1st adaptive method
            'RK12'             : self.RK12,     
            'RK23'             : self.BogackiShampine,
            'RK45'             : self.RK45,
            'Cash-Karp'        : self.CashKarp,
            'Dormand-Prince'   : self.DormandPrince
        }
        
        if method == 'default':
            method = list(self.tableaus.keys())[0]
        
        self.SetIntegrationMethod(method, alpha)
        
    def DisplayMethods(self):
        
        current_method = self.method
        current_alpha  = self.alpha
        
        for key, value in self.tableaus.items():
            self.SetIntegrationMethod(key, .5)
            
            print('{}: {} evaluations and error estimate = {}'.format(key, self.num_evals, self.error_est))
            
        if current_method in self.tableaus:
            self.SetIntegrationMethod(current_method, current_alpha)
            
    # External method allowing user to dynamically change methods
    def SetIntegrationMethod( self, method, alpha = 0.5 ):
        
        self.method = method
        self.alpha  = alpha
        
        if method not in self.tableaus:
            print('Method {} is not valid. Following is a list of valid methods\n'.format(method))
            self.DisplayMethods()
            return
        
        
        # A is matrix of state coefficients (lower diagonal matrix)
        # b is matrix 1 or 2 rows for building solution, 2 is for adaptive
        # c is vector time scalar
        
        # A is square matrix so length of a row of b is length of c
        self.A, self.b = self.tableaus[method](alpha)
        
        self.num_evals = len(self.A)
        self.c = np.zeros(self.num_evals, dtype=float)
        for row in range(0, self.num_evals):
            self.c[row] = sum(self.A[row,:])
            
        self.num_est = len(self.b)
        for row in range(0, self.num_est):
            self.b[row,0] = 1. - sum(self.b[row,1:])
            
        # 1st row of b is the estimate
        # 2nd row of b is the error
        if self.num_est > 1:
            self.error_est = True
            self.b[1,:] -= self.b[0,:]
        else:
            self.error_est = False
            
    def HasErrorEstimate(self):
        return self.error_est
            
        
    # state_obj can provide estimate and covariance information
    # time of measurement should already be in state_obj so it can update
    # f is update function and should contain measuremet and meas cov
    # as well as the measurement transformation function h for this measurement
    # step size is also built into f, so K = h*f()
    #
    # Time Independent implies f does not depend of t directly i.e. f(t, x) == f(x)
    def IntegrationTimeIndependent(self, x0, f):
        
        x = x0.copy()
        K = np.zeros([self.num_evals, len(x0)], dtype=float)
        
        for row in range(0, self.num_evals):
            y = x0.copy()
            for col in range(0,row):
                y += self.A[row,col] * K[col]
            K[row] = f(y)
            x += self.b[0,row] * K[row]
            
        if self.error_est:
            err   = np.zeros(len(x0), dtype=float)
            for row in range(0, self.num_evals):
                err += self.b[1,row] * K[row]
            return x, err
        
        return x
        
        
    # step size is also built into f, so K = h*f()
    def Integration(self, x0, f, t0, t1):
        
        h = t1 - t0
        
        x = x0.copy()
        K = np.zeros([self.num_evals, len(x0)], dtype=float)
        
        for row in range(0, self.num_evals):
            y = x0.copy()
            t = t0 + self.c[row] * h
            for col in range(0,row):
                y += self.A[row,col] * K[col]
            K[row] = f(t, y)
            x += self.b[0,row] * K[row]
            
        if self.error_est:
            err   = np.zeros(len(x0), dtype=float)
            for row in range(0, self.num_evals):
                err += self.b[1,row] * K[row]
            return x, err
        
        return x
    
    
    def ForwardEuler(self, alpha):
        A = np.zeros([1,1], dtype=float)
        b = np.zeros([1,1], dtype=float)
        
        b[0,0] = 1.
        
        return A, b
    
    def Midpoint(self, alpha):
        A = np.zeros([2,2], dtype=float)
        b = np.zeros([1,2], dtype=float)
        
        A[1,0] = .5
        
        return A, b
    
    def Heun2(self, alpha):
        A = np.zeros([2,2], dtype=float)
        b = np.zeros([1,2], dtype=float)
        
        A[1,0] = 1.
        b[0,1] = .5
        
        return A, b
    
    def Ralston2(self, alpha):
        A = np.zeros([2,2], dtype=float)
        b = np.zeros([1,2], dtype=float)
        
        A[1,0] = 2. / 3.
        b[0,1] = .75
        
        return A, b
    
    def Generic2(self, alpha):
        A = np.zeros([2,2], dtype=float)
        b = np.zeros([1,2], dtype=float)
        
        if alpha == 0.:
            print('Warning: Invalid alpha valid for Generic2, changing to alpha = 0.5')
            alpha = .5
        
        A[1,0] = alpha
        b[0,1] = 1. / (2 * alpha)
        
        return A, b
    
    def Kutta3(self, alpha):
        A = np.zeros([3,3], dtype=float)
        b = np.zeros([1,3], dtype=float)
        
        A[1,0] = .5
        A[2,0] = -1.
        A[2,1] = 2.
        b[0,1] = 2./3.
        b[0,2] = 1./6.
        
        return A, b
    
    def Generic3(self, alpha):
        A = np.zeros([3,3], dtype=float)
        b = np.zeros([1,3], dtype=float)
        
        if (alpha == 0.) or (alpha == 2./3.) or (alpha == 1.):
            print('Warning: Invalid alpha valid for Generic3, changing to alpha = 0.5')
            alpha = .5
            
        A[1,0] = alpha
        A[2,0] = 1. + (1. - alpha)/(alpha*(3*alpha-2))
        A[2,1] = -(1. - alpha)/(alpha*(3*alpha-2))
        b[0,1] = 1./(6*alpha*(1-alpha))
        b[0,2] = (2-3*alpha)/(6*alpha*(1-alpha))
        
        return A, b
    
    def Heun3(self, alpha):
        A = np.zeros([3,3], dtype=float)
        b = np.zeros([1,3], dtype=float)
        
        A[1,0] = 1./3.
        A[2,1] = 2./3.
        b[0,2] = .75
        
        return A, b
    
    def Ralston3(self, alpha):
        A = np.zeros([3,3], dtype=float)
        b = np.zeros([1,3], dtype=float)
        
        A[1,0] = .5
        A[2,1] = .75
        b[0,1] = 1./3.
        b[0,2] = 4./9.
        
        return A, b
    
    def SSPRK3(self, alpha):
        A = np.zeros([3,3], dtype=float)
        b = np.zeros([1,3], dtype=float)
        
        A[1,0] = 1.
        A[2,0] = .25
        A[2,1] = .25
        b[0,1] = 1./6.
        b[0,2] = 2./3.
        
        return A, b
    
    def RK4(self, alpha):
        A = np.zeros([4,4], dtype=float)
        b = np.zeros([1,4], dtype=float)
        
        A[1,0] = .5
        A[2,1] = .5
        A[3,2] = 1.
        b[0,1] = 1./3.
        b[0,2] = 1./3.
        b[0,3] = 1./6.
        
        return A, b
    
    def Ralston4(self, alpha):
        A = np.zeros([4,4], dtype=float)
        b = np.zeros([1,4], dtype=float)
        
        A[1,0] = .4
        A[2,0] = .29697761
        A[2,1] = .15875964
        A[3,0] = .21810040
        A[3,1] = -3.05096516
        A[3,2] = 3.83286476
        b[0,1] = -.55148066
        b[0,2] = 1.20553560
        b[0,3] = .17118478
        
        return A, b
    
    # 3/8 rule fourth ordr method
    def Kutta4(self, alpha):
        A = np.zeros([4,4], dtype=float)
        b = np.zeros([1,4], dtype=float)
        
        A[1,0] = 1./3.
        A[2,0] = -1. / 3.
        A[2,1] = 1.
        A[3,0] = 1.
        A[3,1] = -1.
        A[3,2] = 1.
        b[0,1] = 3./8.
        b[0,2] = 3./8.
        b[0,3] = 1./8.
        
        return A, b
    
    
    # Start adaptive methods - b now has two rows
    def HeunEuler(self, alpha):
        A = np.zeros([2,2], dtype=float)
        b = np.zeros([2,2], dtype=float)
        
        A[1,0] = 1.
        b[0,1] = .5
        b[1,1] = 0.
        
        return A, b
    
    def RK12(self, alpha):
        A = np.zeros([3,3], dtype=float)
        b = np.zeros([2,3], dtype=float)
        
        A[1,0] = .5
        A[2,0] = 1./256.
        A[2,1] = 255./256.
        b[0,1] = 255./256.
        b[0,2] = 1./512.
        b[1,1] = 255./256.
        b[1,2] = 0.
        
        return A, b
    
    def BogackiShampine(self, alpha):
        A = np.zeros([4,4], dtype=float)
        b = np.zeros([2,4], dtype=float)
        
        A[1,0] = .5
        A[2,1] = .75
        A[3,0] = 2./9.
        A[3,1] = 1./3.
        A[3,2] = 4./9.
        b[0,1] = 1./3.
        b[0,2] = 4./9.
        b[1,1] = .25
        b[1,2] = 1./3.
        b[1,3] = 1./8.
        
        return A, b
    
    def RK45(self, alpha):
        A = np.zeros([6,6], dtype=float)
        b = np.zeros([2,6], dtype=float)
        
        A[1,0] = .25
        A[2,0] = 3./32.
        A[2,1] = 9./32.
        A[3,0] =  1932./2197.
        A[3,1] = -7200./2197.
        A[3,2] =  7296./2197.
        A[4,0] = 439./216.
        A[4,1] = -8.
        A[4,2] = 3680./513.
        A[4,3] = -845./4104.
        A[5,0] = -8./27.
        A[5,1] = 2.
        A[5,2] = -3544./2565.
        A[5,3] = 1859./4104.
        A[5,4] = -11./40.
        b[0,2] = 6656./12825.
        b[0,3] = 28561./56430.
        b[0,4] = -9./50.
        b[0,5] = 2./55.
        b[1,2] = 1408./2565.
        b[1,3] = 2197./4104.
        b[1,4] = -.2
        
        return A, b
    
    def CashKarp(self, alpha):
        A = np.zeros([6,6], dtype=float)
        b = np.zeros([2,6], dtype=float)
        
        A[1,0] = .2
        A[2,0] = 3./40.
        A[2,1] = 9./40.
        A[3,0] = .3
        A[3,1] = -.9
        A[3,2] = 1.2
        A[4,0] = -11./54.
        A[4,1] = 2.5
        A[4,2] = -70./27.
        A[4,3] = 35./27.
        b[0,2] = 250./621.
        b[0,3] = 125./594.
        b[0,5] = 512./1771.
        b[1,2] = 18575./48384.
        b[1,3] = 13525./55296.
        b[1,4] = 277./14336.
        b[1,5] = .25
        
        return A, b
    
    def DormandPrince(self, alpha):
        A = np.zeros([7,7], dtype=float)
        b = np.zeros([2,7], dtype=float)
        
        A[1,0] = .2
        A[2,0] = 3./40.
        A[2,1] = 9./40.
        A[3,0] = 44./45.
        A[3,1] = -56./15.
        A[3,2] = 32./9.
        A[4,0] = 19372./6561.
        A[4,1] = -25360./2187.
        A[4,2] = 64448./6561.
        A[4,3] = -212./729.
        A[5,0] = 9017./3168.
        A[5,1] = -355./33.
        A[5,2] = 46732./5247.
        A[5,3] = 49./176.
        A[5,4] = -5103./18656.
        A[6,0] = 35./384.
        A[6,2] = 500./1113.
        A[6,3] = 125./192.
        A[6,4] = -2187./6784.
        A[6,5] = 11./84.
        b[0,2] = 500./1113.
        b[0,3] = 125./192.
        b[0,4] = -2187./6784.
        b[0,5] = 11./84.
        b[1,2] = 7571./16695.
        b[1,3] = 393./640.
        b[1,4] = -92097./339200.
        b[1,5] = 187./2100.
        b[1,6] = 1./40.
        
        return A, b