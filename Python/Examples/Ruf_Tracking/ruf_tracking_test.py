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
import matplotlib.pyplot as plt


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

    save_state_est = np.zeros([len(measurement_array), 6])
    save_measurement_array = np.zeros([len(measurement_array), 3])
    save_time = np.zeros(len(measurement_array))
    
    meas_count = 0
    
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
        
        
        save_time[meas_count] = time
        save_measurement_array[meas_count] = meas.Truth
        save_state_est[meas_count] = state_est
        
        meas_count = meas_count + 1
    
    plt.close('all')
    plt.figure()
    plt.subplot(211)
    plt.plot(save_time, save_measurement_array[:,0], 'r-', label='X Truth')
    plt.plot(save_time, save_state_est[:,0], 'b-', label='X Est')
    plt.xlabel('Time')
    plt.ylabel('Meters')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(save_time, save_state_est[:,0] - save_measurement_array[:,0], 'g-', label='X Error')
    plt.xlabel('Time')
    plt.ylabel('Meters')
    
    plt.figure()
    plt.subplot(211)
    plt.plot(save_time, save_measurement_array[:,2], 'r-', label='Z Truth')
    plt.plot(save_time, save_state_est[:,2], 'b-', label='Z Est')
    plt.xlabel('Time')
    plt.ylabel('Meters')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(save_time, save_state_est[:,2] - save_measurement_array[:,2], 'g-', label='Z Error')
    plt.xlabel('Time')
    plt.ylabel('Meters')
    
      
    