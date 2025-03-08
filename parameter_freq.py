# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 2023

@author: ase99
"""
# %% ------------------------------------------------------------------------
import numpy as np

# extract the estimated values from the freq_loop_PP.py files to visualize them aafo z_min. The includeLogV option is used by bayes_PP.py to generate the priors.

def Timescape(timescape, includeLogV=False):
    
    TS_omega = timescape[0]
    TS_alpha = timescape[1]
    TS_x = timescape[2]
    TS_beta = timescape[4]
    TS_c = timescape[5]
    TS_M = timescape[7]
    TS_omega = np.transpose(TS_omega)
    TS_alpha = np.transpose(TS_alpha)
    TS_x = np.transpose(TS_x)
    TS_beta = np.transpose(TS_beta)
    TS_c = np.transpose(TS_c)
    
    if includeLogV:
        TS_fv = 0.5*(np.sqrt(9.-8.*TS_omega)-1) # fv0 = 0.5*(np.sqrt(9.-8.*om)-1) # OM = 0.5*(1.-fv0)*(2.+fv0)
        V_x = np.transpose(timescape[3])
        V_c = np.transpose(timescape[6])
        V_M = np.transpose(timescape[8])
        if includeLogV == 1:
            return TS_fv, TS_alpha, TS_x, np.log10(V_x), TS_beta, TS_c, np.log10(V_c), TS_M, np.log10(V_M)
        elif includeLogV == 2:
            return TS_omega, TS_alpha, TS_x, V_x, TS_beta, TS_c, V_c, TS_M, V_M
        elif includeLogV == 3:
            return TS_fv, TS_alpha, TS_x, V_x, TS_beta, TS_c, V_c, TS_M, V_M
    
    
    return TS_omega, TS_alpha, TS_beta, TS_x, TS_c, TS_M

def LCDM(lamCDM, includeLogV=False):
    
    lamCDM_omega = lamCDM[0]
    lamCDM_alpha = lamCDM[2]
    lamCDM_x = lamCDM[3]
    lamCDM_beta = lamCDM[5]
    lamCDM_c = lamCDM[6]
    lamCDM_M = lamCDM[8]
    lamCDM_omega = np.transpose(lamCDM_omega)
    lamCDM_x = np.transpose(lamCDM_x)
    lamCDM_alpha = np.transpose(lamCDM_alpha)
    lamCDM_beta = np.transpose(lamCDM_beta)
    lamCDM_c = np.transpose(lamCDM_c)
    
    if includeLogV:
        V_x = np.transpose(lamCDM[3])
        V_c = np.transpose(lamCDM[6])
        V_M = np.transpose(lamCDM[8])
        if includeLogV == 1:
            return lamCDM_omega, lamCDM_alpha, lamCDM_x, np.log10(V_x), lamCDM_beta, lamCDM_c, np.log10(V_c), lamCDM_M, np.log10(V_M)
        elif includeLogV == 2 or includeLogV == 3:
            return lamCDM_omega, lamCDM_alpha, lamCDM_x, V_x, lamCDM_beta, lamCDM_c, V_c, lamCDM_M, V_M
    
    return lamCDM_omega, lamCDM_alpha, lamCDM_beta, lamCDM_x, lamCDM_c, lamCDM_M

def LTA(lta_params, includeLogV=False):
    """
    Extract parameters from LTA model results
    """
    LTA_omega = lta_params[0]
    LTA_sf = lta_params[1]  # Scale factor parameter
    LTA_alpha = lta_params[2]
    LTA_x = lta_params[3]
    LTA_beta = lta_params[5]
    LTA_c = lta_params[6]
    LTA_M = lta_params[8]
    
    LTA_omega = np.transpose(LTA_omega)
    LTA_sf = np.transpose(LTA_sf)
    LTA_alpha = np.transpose(LTA_alpha)
    LTA_x = np.transpose(LTA_x)
    LTA_beta = np.transpose(LTA_beta)
    LTA_c = np.transpose(LTA_c)
    
    if includeLogV:
        V_x = np.transpose(lta_params[4])
        V_c = np.transpose(lta_params[7])
        V_M = np.transpose(lta_params[9])
        if includeLogV == 1:
            return LTA_omega, LTA_sf, LTA_alpha, LTA_x, np.log10(V_x), LTA_beta, LTA_c, np.log10(V_c), LTA_M, np.log10(V_M)
        elif includeLogV == 2:
            return LTA_omega, LTA_sf, LTA_alpha, LTA_x, V_x, LTA_beta, LTA_c, V_c, LTA_M, V_M
    
    return LTA_omega, LTA_sf, LTA_alpha, LTA_beta, LTA_x, LTA_c, LTA_M

def Milne(milne, includeLogV=False):
    
    milne_omega = milne[0]
    milne_alpha = milne[2]
    milne_x = milne[3]
    milne_beta = milne[5]
    milne_c = milne[6]
    milne_M = milne[8]
    milne_omega = np.transpose(milne_omega)
    milne_x = np.transpose(milne_x)
    milne_alpha = np.transpose(milne_alpha)
    milne_beta = np.transpose(milne_beta)
    milne_c = np.transpose(milne_c)
    
    if includeLogV:
        V_x = np.transpose(milne[3])
        V_c = np.transpose(milne[6])
        V_M = np.transpose(milne[8])
        if includeLogV == 1:
            return milne_omega, milne_alpha, milne_x, np.log10(V_x), milne_beta, milne_c, np.log10(V_c), milne_M, np.log10(V_M)
        elif includeLogV == 2 or includeLogV == 3:
            return milne_omega, milne_alpha, milne_x, V_x, milne_beta, milne_c, V_c, milne_M, V_M
    
    return milne_omega, milne_alpha, milne_beta, milne_x, milne_c, milne_M
