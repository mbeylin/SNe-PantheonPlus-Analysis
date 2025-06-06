import sys
import pandas as pd
import numpy as np
allnegew = {}
ewByFile = True # whether to calculate the eigenvalues within PPcovFIT (True) or at the end (False). Only if True all non-positive definite eigenvalues are dropped
mkposdef = True # force the covariance matrix to be positive definite by dropping the bad SNe. These SNe latter are printed at the end
nominal = 'FITOPT000' # file used to get the statistical covariance and the reference values
nominalmu = 'MUOPT000'
reducelowz = False # whether to reduce the excess of low-redshift SNe to an amount similar to the JLA/P+ subsample
Nseeds = 0 # number of random subsamples to be drawn
m = 1000 # size of the random subsamples
choosewithoutreduce = 'high' # for 'high', the third with highest zCMB is fully taken and the other two are chosen randomly
# other possibilities: 'low' (the other way round) or 'none' (usual random subsample with same distribution as the full one)

###################### DEFINITIONS ###################################################
def extractsne(name):
    return name.rsplit('_', maxsplit=1)[0]

def stackidxdf(df): # reduce multiindex of df to a single-level index
    df.index = np.array(['_'.join(col) for col in df.index])
    return df

def blockidx(indices): # indices of covariance matrix for given SNe
    form = pd.DataFrame(np.zeros((len(indices), 3)), index = indices, columns = ['mB', 'x1', 'zc'])
    return np.array(['_'.join(col) for col in form.stack().index])

def blocksne(indices): # reverse of blockidx
    return [extractsne(s) for s in indices[::3]]

def jacentry(s, fulldf): # only non-trivial entry of Jacobian block matrices
    return -2.5 / np.log(10) / fulldf.iloc[s].loc['x0']

def eigenvalues(covdf, fitopt, allsne, returnsne=True):
    ew = np.linalg.eigvals(covdf)
    print(fitopt, ':', len(ew[ew < 0]), 'out of', len(ew), 'are less than 0, ', len(ew[ew == 0]), 'are equal to zero')
    
    if returnsne:
        alluniquesne = np.unique([extractsne(s) for s in allsne])
        sneevent = {sne: [event for event in allsne if extractsne(event) == sne] for sne in alluniquesne}
        sneew = {sne: np.linalg.eigvals(covdf.loc[blockidx(sneevent[sne]), blockidx(sneevent[sne])]) for sne in sneevent.keys()}
        negew = np.concatenate([sneevent[sne] for sne in sneew.keys() if not np.all(sneew[sne] >= 0)])
        allnegew[fitopt] = negew

def magmagmatrix(magdf): # return matrix with only the (mB, mB) entries of the diagonal being non-zero
    idxcov = blockidx(magdf.index)
    empty = pd.DataFrame(np.zeros(len(magdf.index)), index=magdf.index) # DataFrame with all zero entries (for x1 and c diagonal components of the covariance)
    table = pd.concat([magdf] + [empty]*2, axis=1)
    matrix = pd.DataFrame(np.diagflat(np.array(table.stack())))
    matrix.columns = idxcov
    matrix.index = idxcov
    return matrix

def offdiagduplicates(diagdf):
    offdiagdf = diagdf.copy()
    for i in range(int(offdiagdf.shape[0] / 3)):
        for j in range(i):
            if np.all(offdiagdf.index[(3*i):(3*i+3)] == offdiagdf.index[(3*j):(3*j+3)]):
                subdfi = offdiagdf.iloc[(3*i):(3*i+3), (3*i):(3*i+3)]
                subdfj = offdiagdf.iloc[(3*j):(3*j+3), (3*j):(3*j+3)]
                offdiagdf.iloc[(3*j):(3*j+3), (3*i):(3*i+3)] = (subdfi + subdfj) / 2
                offdiagdf.iloc[(3*i):(3*i+3), (3*j):(3*j+3)] = (subdfi + subdfj) / 2
    return offdiagdf

# -------------------- generate 3N x 3N covariance matrix ----------------------------
def PPcovFIT(fulldf, fitopt):
    fulldf = fulldf.sort_index()
    jac = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                     np.diagflat([jacentry(s, fulldf), 1, 1]), # Jacobi matrix
                     np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covdiag = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                         np.diagflat(np.array(fulldf.iloc[s].loc[['x0ERR', 'x1ERR', 'cERR']])**2), # diagonal of cov
                         np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covlow = np.block([[np.zeros((3, s*3)), # zero blocks in same row before cov block
                        np.array([[0]*3, [fulldf.iloc[s].loc['COV_x1_x0']] + [0]*2, [fulldf.iloc[s].loc['COV_c_x0'], fulldf.iloc[s].loc['COV_x1_c'], 0]]), # cov matrix from PP for (x0, x1, c)
                        np.zeros((3, (len(fulldf.index) - s-1)*3))] for s in range(len(fulldf.index))]) # zero blocks in same row after cov block
    covx0 = covlow + covdiag.astype(float) + np.transpose(covlow) # cov matrix for all SNe events
    covdf = pd.DataFrame(np.dot(jac, np.dot(covx0, np.transpose(jac))))
    idxcov = blockidx(fulldf.index)
    covdf.columns = idxcov
    covdf.index = idxcov
        
    eigenvalues(covdf, fitopt, fulldf.index, ewByFile)
    
    return covdf

def PPcovDUPL(fulldf):
    # Create a copy with only needed columns
    dupldf = fulldf.loc[:, ['mB', 'x1', 'c', 'CID', 'IDSURVEY']].sort_index().copy()
    
    # Convert numeric columns explicitly before any operations
    numeric_columns = ['mB', 'x1', 'c']
    for col in numeric_columns:
        dupldf[col] = pd.to_numeric(dupldf[col], errors='coerce')
    
    # Keep CID as string to preserve leading zeros
    dupldf['CID'] = dupldf['CID'].astype(str)
    dupldf['IDSURVEY'] = dupldf['IDSURVEY'].astype(str)
    
    # Calculate means
    mean_values = pd.DataFrame()
    for col in numeric_columns:
        mean_values[col] = dupldf.groupby('CID')[col].transform('mean')
    
    # Calculate residuals while keeping CID and IDSURVEY
    residual = dupldf.copy()
    residual[numeric_columns] = dupldf[numeric_columns].subtract(mean_values)
    
    # Now set the index
    residual = residual.set_index(['CID', 'IDSURVEY'], append=True)
    
    # Drop the unnecessary columns and levels
    residual = residual[numeric_columns]
    residual = residual.droplevel(['CID', 'IDSURVEY'])
    
    # Stack and process
    orgdelta = pd.DataFrame(residual.stack())
    orgdelta.index = blockidx(residual.index)
    
    # Process each supernova
    sigmae = []
    unique_cids = dupldf['CID'].unique()
    
    for sn in unique_cids:
        snversions = [sv for sv in residual.index if extractsne(sv) == sn]
        if len(snversions) > 0:
            delta = orgdelta.loc[blockidx(snversions), :]
            sigma = delta @ delta.T
            const = magmagmatrix(pd.DataFrame(np.full(int(len(sigma.index) / 3), 0.102), index=snversions))
            const = offdiagduplicates(const) - const
            sigmae.append(sigma + const)
    
    # Combine results
    sigmadupl = pd.concat(sigmae).fillna(0)
    return sigmadupl

def PPcovSYST(fulldf, comparedf, scale): # systematic covariance according to Eq. (7) of Brout et al. 2022 (arXiv:2202.04077)
    fulldf = fulldf.sort_index() # mu
    val = pd.DataFrame(fulldf.loc[:, ['mB', 'x1', 'c']].stack())
    val.index = blockidx(fulldf.index)
    comparedf = comparedf.sort_index() # reference value of delta mu
    ref = pd.DataFrame(comparedf.loc[:, ['mB', 'x1', 'c']].stack())
    ref.index = blockidx(comparedf.index)
    delta = (val - ref) * scale # d delta mu / d S * sigma
    systdf = delta @ delta.T
    return systdf # return covariance contribution of a single psi

def PPcovSTAT(fulldf):
    fulldf = fulldf.sort_index()
    # sigma_lens^2 = (0.055 * boostz)^2
    sigma2lens = pd.DataFrame((0.055 * boostz(fulldf.zHEL, fulldf.RA, fulldf.DEC))**2)
    
    # sigma_z^2 = D_boostz**2
    # D_boostz = d/dz boostz(z) * D_z = (1 + (vel/C)*costheta) * D_z = boostz(D_z)
    sigma2z = pd.DataFrame(boostz(fulldf.zHELERR, fulldf.RA, fulldf.DEC)**2)
        
    # 3N x 3N matrix
    sigma2 = offdiagduplicates(magmagmatrix(sigma2z + sigma2lens)) # populate entries with same SNe (with and without same survey)
    return sigma2

# -------------------- correct zCMB --------------------------------------------------
def boostz(z, RAdeg, DECdeg, vel=371.0, RA0=168.0118667, DEC0=-6.98303424):
    '''
    Tully et al 2008. Tully et al reference everything in the local group frame and because we are transforming to the local group frame, therefore we use this instead of PLANCK2013 data
    
    vcmb = 371.0 # km/s #Velocity boost of CMB
    l_cmb = 264.14 # CMB multipole direction (degrees)
    b_cmb = 48.26 # CMB multipole direction (degrees)
    # converts to
    ra_cmb = 168.0118667 #Right Ascension of CMB
    dec_cmb = -6.98303424 #Declination of CMB
    '''
    # Angular coords should be in degrees and velocity in km/s
    C = 2.99792458e5 # km/s #Light
    RA = np.radians(RAdeg)
    DEC = np.radians(DECdeg)
    RA0 = np.radians(RA0)
    DEC0 = np.radians(DEC0)
    costheta = np.sin(DEC)*np.sin(DEC0) + np.cos(DEC)*np.cos(DEC0)*np.cos(RA-RA0) 
    return z + (vel/C)*costheta*(1+z)

# -------------------- select columns as in BuildJLACases ----------------------------
def PPinput(fulldf):
    myinput = fulldf.loc[:, ['zCMB', 'mB', 'x1', 'c', 'HOST_LOGMASS', 'IDSURVEY', 'zHEL', 'RA', 'DEC', 'x1ERR', 'cERR', 'COV_x1_c']]
    myinput.zCMB = boostz(fulldf.zHEL, fulldf.RA, fulldf.DEC)
    return myinput.sort_index()

###################### RUN ###########################################################
# -------------------- read scaling of files -----------------------------------------
summary = pd.read_csv('Pantheon/fitopts_summary.csv', index_col=0)
summary.loc[:, 'weights'] = summary.loc[:, 'scale'] * summary.loc[:, 'vpecTo0']
summarymu = pd.read_csv('Pantheon/muopts_summary.csv', index_col=0)

# -------------------- read files ----------------------------------------------------
print('read files')
dfs = {fo: stackidxdf(pd.read_table('Pantheon/calibration_files/' + fo + '_MUOPT000.FITRES', sep=' ', comment='#', skipinitialspace=True, index_col=0).dropna(how='all', axis=0).dropna(how='all', axis=1).set_index(['CID', 'IDSURVEY'], append=True, drop=False).droplevel('VARNAMES:', axis=0).rename(str, axis='index')) for fo in summary.index}
dms = {mo: stackidxdf(pd.read_table('Pantheon/calibration_files/FITOPT000_' + mo + '.FITRES', sep=' ', comment='#', skipinitialspace=True, index_col=0).dropna(how='all', axis=0).dropna(how='all', axis=1).set_index(['CID', 'IDSURVEY'], append=True, drop=False).droplevel('VARNAMES:', axis=0).rename(str, axis='index')) for mo in summarymu.index}

# -------------------- calculate PPcov and PPinput for individual files --------------
print('input calculation for ' + nominal, end=' ')
inp = PPinput(dfs[nominal])
print(inp.shape)
print('start covariance calculation of fit for ' + nominal, end=' ')
covfit = PPcovFIT(dfs[nominal], nominal)
print(covfit.shape)
print('start covariance calculation for duplicated SNe', end=' ')
covdupl = PPcovDUPL(dfs[nominal])
print(covdupl.shape)
print('start statistical covariance calculation', end=' ')
covstat = PPcovSTAT(dfs[nominal])
print(covstat.shape)
print('start systematic covariance calculation from FITOPTs')
covsyst = [PPcovSYST(dfs[fo], inp, summary.loc[fo, 'weights']) for fo in np.setdiff1d(summary.index, [nominal])]
allcovsyst = pd.concat(covsyst, axis=1).groupby(level=0, axis=1).sum()
print('start systematic covariance calculation from MUOPTs')
covmu = [PPcovSYST(dms[mo], inp, summarymu.loc[mo, 'scale']) for mo in np.setdiff1d(summarymu.index, [nominalmu])]
allcovmu = pd.concat(covmu, axis=1).groupby(level=0, axis=1).sum()

# -------------------- combine PPcov and PPinput -------------------------------------
print('combine')
allinp = inp
allcov = pd.concat([allcovsyst, allcovmu, covfit, covdupl, covstat], axis=1).groupby(level=0, axis=1).sum() # sum over all psi of Eq. (7) to get full covariance, add statistical & duplication covariances
colinp = list(inp.columns)
idxinp = list(inp.index)
eigenvalues(allcov, 'combined', allinp.index, returnsne = not ewByFile)
if mkposdef:
    idxinp = np.setdiff1d(idxinp, np.unique(np.concatenate([allnegew[k] for k in allnegew.keys()])))
allinp = allinp.loc[idxinp, colinp]
idxcov = blockidx(idxinp)
colcov = idxcov
allcov = allcov.loc[idxcov, colcov]
eigenvalues(allcov, 'finalcov', allinp.index, returnsne=False) # check if resulting matrix has negative eigenvalues

versionname = '1690'
if Nseeds == 0:
    # -------------------- reduce low-redshift population --------------------------------
    if reducelowz:
        highz = allinp.loc[allinp.zCMB >= 5e-2].index
        lowz = allinp.loc[allinp.zCMB < 5e-2].index
        jla = np.array(pd.read_csv('Pantheon/joinedsample_CID+IDSURVEY.csv', index_col=0).T)[0]
        subinp = allinp.loc[np.intersect1d(jla, allinp.index)]
        frac = (subinp.loc[subinp.zCMB < 5e-2].shape[0] / subinp.shape[0]) / (len(lowz) / allinp.shape[0])
        allinp = pd.concat([allinp.loc[lowz].sample(frac = frac), allinp.loc[highz]])
        idxcov = blockidx(allinp.index)
        colcov = idxcov
        allcov = allcov.loc[idxcov, colcov]

    # -------------------- save ----------------------------------------------------------
    print('save', versionname)
    np.savetxt('Pantheon/Build/PP_' + versionname + '_COVd.txt', np.array(allcov))
    allinp.to_csv('Pantheon/Build/PP_' + versionname + '_input.csv')
    np.savetxt('Pantheon/Build/PP_' + versionname + '_input.txt', 
               np.array(allinp.loc[:, ['zCMB', 'mB', 'x1', 'c', 'HOST_LOGMASS', 'IDSURVEY', 'zHEL', 'RA', 'DEC']]))
    print('resulting shape:', allcov.shape, allinp.shape)
    print(allnegew)

alluniquesne = np.unique([extractsne(s) for s in allinp.sort_values(by='zCMB', ascending=True).index])
ll = 0 # the size of the sample of SNe that is taken in full
if choosewithoutreduce == 'high' or choosewithoutreduce == 'low':
    ll = int(m / 3)
l = len(alluniquesne) - ll # the number of SNe that are chosen randomly
print(l)
idx = np.arange(l)
chooseall = np.array([]) # the indices of the sample of SNe that is taken in full
if choosewithoutreduce == 'low':
    idx = idx + ll
    chooseall = np.arange(ll)
elif choosewithoutreduce == 'high':
    chooseall = l + np.arange(ll)
p = np.full(l, 1/l)
versionname = '1690random' + str(m) + choosewithoutreduce
for seed in range(Nseeds):
    np.random.seed()
    choice = np.unique(np.random.choice(idx, size = m, replace=False, p=p))
    choice = np.append(choice, chooseall)
    choicesne = np.concatenate([[sv for sv in allinp.index if extractsne(sv) == sn] for sn in np.array(alluniquesne)[choice]])
    print(choicesne)
    
    dfsamp = allinp.loc[choicesne]
    print(allinp.shape, dfsamp.shape)
    dfcov = allcov.copy().loc[blockidx(dfsamp.index), blockidx(dfsamp.index)]

    # -------------------- save ----------------------------------------------------------
    print('save', versionname, seed)
    np.savetxt('Pantheon/Build/PP_' + versionname + '_' + str(seed) + '_COVd.txt', np.array(dfcov))
    np.savetxt('Pantheon/Build/PP_' + versionname + '_' + str(seed) + '_input.txt', np.array(dfsamp))
    print('resulting shape:', dfcov.shape, dfsamp.shape)
    print(allnegew)
