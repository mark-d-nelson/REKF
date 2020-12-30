# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:06:11 2020

@author: Mark
"""

import sys 
sys.path.append('..')

import numpy as np
import ButcherTableau as BT

h = .5
b = .25

def QuadFunc(x):
    return h * x

bt = BT.ButcherTableauExplicitMethods('RK45')

x = np.asarray( [b] )
y = bt.IntegrationTimeIndependent(x, QuadFunc)

print(np.exp( np.log(b) + h ))
print(y[0])