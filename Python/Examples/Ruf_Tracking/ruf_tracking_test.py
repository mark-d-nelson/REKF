#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:58:18 2020

@author: megan_nelson
"""

import sys 
sys.path.append('../..')

import ButcherTableau as BT
import KalmanGain     as KG
import load_ruf_tracking_data as lrtd
import ruf_tracking_function as rufunction
import numpy as np


if __name__ == '__main__':
    
    Q = np.zeros([6, 6])
    Q[3:6, 3:6] = np.eye(3) * 1e-9
    
    bt = BT.ButcherTableauExplicitMethods('RK45')
    pbt = BT.ButcherTableauExplicitMethods('RK4')
    kg = KG.KalmanGain( rufunction.TemporalDerivative, pbt, 6, [] )
    
    # load measurement array
    measurement_array = lrtd.GetMeasurements()
    
    time       = 0.
    state_est  = np.asarray( [100., 0., 5., 0., 0., -.05] )
    covariance = np.zeros([6, 6])
    covariance[0:3, 0:3] = np.eye(3) * 100.
    covariance[3:6, 3:6] = np.eye(3) * 0.05 * 0.05

    
    for meas in measurement_array:
        
        state_est = kg.SetGainParam( time, state_est, covariance, meas )
        
        if bt.HasErrorEstimate():
            state_est, err = bt.IntegrationTimeIndependent(state_est, kg.GetChangeInState)
        else:
            state_est = bt.IntegrationTimeIndependent(state_est, kg.GetChangeInState)
            err = []
            
        covariance = kg.UpdateCovariance( state_est )
        if len(err) > 0:
            covariance += np.outer(err, err)
        
        time = meas.Time
        print('Estimate {}: {}'.format(time, state_est))
        