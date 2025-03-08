from __future__ import division
from scipy import interpolate
import numpy as np

class spl(object):
    def __init__(self, name, Ntotal):
        self.name = name
        self.Ntotal = Ntotal

        interp = np.load('Pantheon/Build/PP_' + self.name + '_tabledL_ts.npy') # e.g. self.name = 'JLA_Common/JLA'
        oms = np.linspace(0.00,0.99,100)
        oms[0] = 0.001
        self.ts = []
        for i in range(self.Ntotal):
            self.ts.append(interpolate.InterpolatedUnivariateSpline(oms, interp[i]))
    
        interpu = np.load('Pantheon/Build/PP_' + self.name + '_tabledL_lcdm.npy')
        oms = np.linspace(0,1.0,101)
        ols = np.linspace(0,1.0,101)
        self.lcdm = []
        for i in range(self.Ntotal):
            self.lcdm.append(interpolate.RectBivariateSpline(oms, ols, interpu[i]))
        
        # Load LTA interpolation
        interp_lta = np.load('Pantheon/Build/PP_' + self.name + '_tabledL_lta.npy')
        oms = np.linspace(0.00, 0.99, 100)
        oms[0] = 0.001
        sfs = np.linspace(0.979, 0.983, 100)
        self.lta = []
        for i in range(self.Ntotal):
            self.lta.append(interpolate.RectBivariateSpline(oms, sfs, interp_lta[i]))