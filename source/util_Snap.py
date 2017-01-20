import numpy as np
from scipy.special import factorial

'''
Numerical utility
'''


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

'''
File utility
'''


def stability_checker(homedir, filename, truetheta):
    dat = load_dat(homedir, filename)

    match = np.zeros([1, len(truetheta)])
    for k in range(len(truetheta)):
        #print "true: "+ str(truetheta[k])
        if truetheta[k] >  0:
            match[0][k] = np.sum(np.abs((dat[:,k] - truetheta[k])/ truetheta[k]) < tolerance)
            #print match[0][k]
        else:
            match[0][k] = np.sum(np.abs((dat[:,k] - truetheta[k])) < tolerance)
            #print match[0][k]

    return match

def all_stability_checker(homedir, filename, truetheta):
    dat = load_dat(homedir, filename)
    Ndat = dat.shape[0]
    matches = np.ones(Ndat)

    for k in range(len(truetheta)):
        #print "true: "+ str(truetheta[k])
        if truetheta[k] >  0:
            matches = matches* (   np.abs((dat[:,k] - truetheta[k])/ truetheta[k]) < tolerance)
            #print match[0][k]
        else:
            matches = matches * (np.abs((dat[:,k] - truetheta[k])) < tolerance)
            #print match[0][k]

    return matches

'''
loads file from a filenamepath
'''

def load_dat(homedir, filename):
    filepath = os.path.join(homedir, "records", filename)
    dat = pickle.load( open( filepath, "rb" ) )
    return dat

'''
returns dictionary of arrays for dictionary of filenames
'''
def load_arrays(filenames):
    dats = {}
    for key in filenames.keys():
    dats.update({key:load_dat(homedir, filenames[key])}  )


'''
Make two overlaced histrogram for a pair of snapshots
'''

def compare_snaps(times, nxs, theta_test, theta_truth= np.array([  0. ,   0.5,  25. ,   6. ,   0.2]) ):

    dsystemTruth = dd.Discrete_Doucet_system(theta = theta_truth)

    dsystem2 =dd.Discrete_Doucet_system(theta = theta_test)


    snapshotsTruth= dsystemTruth.make_snapshots(nxs, times, np.array([-1.5]))
    snapshots2= dsystem2.make_snapshots(nxs, times, np.array([-1.5]))
    plt.figure(figsize=(20, 4))

    for k in range(len(times)):
        plt.subplot(1,len(times), k +1 )
        plt.title('t=' + str(times[k]))

        plt.hist(snapshotsTruth[times[k]], alpha = 0.5, bins = 50)
        plt.hist(snapshots2[times[k]], alpha = 0.5, bins = 50)

    plt.show()
