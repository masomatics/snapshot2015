__author__ = 'markov'


import numpy as np
import scipy as sp
from datetime import datetime
from timeit import default_timer as timer
import sys
import pickle

class Numerical_test:

    def __init__(self,  alpha, theta_init):

        self.alpha = alpha
        self.theta_init = theta_init


    def compare(self, xdat, xobs, sigma):

        Nx_obs = len(xobs)
        Nx_dat = len(xdat)

        comp_array = np.array( [list(np.array(xdat))] * Nx_obs)
        comp_array2 = np.transpose(np.array([xobs] * Nx_dat), [1,0,2])

        diffsqr = np.sum(np.power(np.abs(comp_array - comp_array2) /sigma, 2)/2, axis = 2)

        return diffsqr

    def compare_snap(self, xdat, snapshot, sigma, observed=[np.array([0])], likelihood = False):

        Nx_obs = len(snapshot[0][1])
        Nx_dat = len(xdat)

        diffsqr = np.zeros([Nx_obs, Nx_dat])
        log_likelihood = 0


        for obs_idx in range(0,len(observed)):

            dim_interest = observed[obs_idx]
            #print dim_interest
            #print sigma[dim_interest]
            diffsqr = diffsqr + self.compare(xdat[:, dim_interest], snapshot[obs_idx][1], sigma = sigma[dim_interest])

        pyx = np.exp(-(diffsqr - np.transpose(np.matrix([[np.min(row) for row in diffsqr]] * Nx_dat))))
        pyxm = np.matrix([row / np.sum(row) for row in pyx.tolist()])

        print xdat
        print snapshot
        print pyxm
        print diffsqr
        print np.multiply(pyxm , diffsqr)
        log_likelihood = -np.sum(np.multiply(pyxm , diffsqr))
        px_new = np.array(np.sum(pyxm, 0))[0]
        px_new = px_new / sum(px_new)

        if likelihood:
            return px_new, log_likelihood
        else:
            return px_new



    def initialize(self, Nx, init_snap, seed = 1, prob = []):

        #This initialize is different from the original in that it takes in the distribution
        np.random.seed(seed)
        numsample = init_snap.shape[0]   #number of supports
        indices = range(0, numsample)
        if len(prob) == 0 :
            x0 = np.random.choice(indices, Nx)
        else:
            x0 = np.random.choice(indices, Nx, p = prob)


        return init_snap[x0,:]