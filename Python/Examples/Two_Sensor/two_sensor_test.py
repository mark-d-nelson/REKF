#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 21:43:43 2020

@author: Mark
"""

import sys 
sys.path.append('../..')

import ButcherTableau as BT
import KalmanGain     as KG
import load_two_sensor_data as ltsd
import numpy as np


if __name__ == '__main__':
    
    bt = BT.ButcherTableauExplicitMethods('RK45')
    kg = KG.KalmanGain( [], 2, [] )
    
    # load measurement array
    measurement_array = ltsd.GetMeasurements()
    
    time       = 0.
    state_est  = np.asarray( [.5, .1] )
    covariance = np.eye(2) * 100.
    
    for meas in measurement_array:
        kg.SetGainParam( time, state_est, covariance, meas )
        
        if bt.HasErrorEstimate():
            state_est, err = bt.IntegrationTimeIndependent(state_est, kg.GetChangeInState)
        else:
            state_est = bt.IntegrationTimeIndependent(state_est, kg.GetChangeInState)
            err = []
            
        covariance = kg.UpdateCovariance( state_est )
        if not err:
            covariance += np.outer(err, err)
            
        time = meas.Time