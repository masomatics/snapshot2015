import numpy as np
from scipy.special import factorial


def normalize_logP(logp):
    """
    normalize the lop and returns the normalized exp(logp) probability.
    """
    normalizer = np.max(logp)
    normal_logp = logp - normalizer
    prob =  np.exp(normal_logp)
    normal_prob = prob / np.sum(prob)
    return normal_prob

def log_factorial_singlearray(ks):
    return np.array([sum(np.log(range(1,np.int(k)+1))) for k in ks])

def log_factorial(k_array):
    fac_array = np.zeros([k_array.shape[0], k_array.shape[1]])
    for k in range(k_array.shape[0]):
        fac_array[k] = log_factorial_singlearray(k_array[k])
    return fac_array

def logP_Path(xPath, muPath):
    vals = np.exp(-muPath) * muPath**xPath / factorial(xPath)
    return np.sum(np.log(vals))

def Ps_Path(xPath, muPath):
    vals = np.exp(-muPath) * muPath**xPath / factorial(xPath)
    return vals

def logPsPath(xPath, muPath):
    #print xPath, muPath
    logvals = -muPath + xPath *np.log(muPath) - log_factorial_singlearray(xPath)
    #print np.log(muPath), muPath, xPath
    logvals[np.where(np.isnan(logvals))[0]] = 0.

    return logvals

def stim_gen(prevstim, N):
    stim =  np.random.poisson(prevstim, N)
    return stim_gen

def stim_gen_tau(prevstim, N , tau, sigma = 2):
    stimpaths = np.zeros([N, tau])
    logPpaths = np.zeros(N)
    stim_old = np.array([prevstim] * N).reshape([1,N])
    for k in range(tau):

        nextstim =  np.abs(stim_old +  np.floor( np.random.normal(0,sigma, [1,N]) ))
        stimpaths[:, k] = nextstim
        stim_old = stimpaths[:, k]

    return stimpaths
