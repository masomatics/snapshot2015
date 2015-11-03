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


    def compare(self, xdat, xobs, sigma, observed = [np.array([0])]):

        Nx_obs = len(xobs)
        Nx_dat = len(xdat)

        diffsqr = np.zeros([Nx_obs, Nx_dat])
        for obs_idx in observed:
            comp_array = np.array( [list(np.array(xdat[:, obs_idx]))] * Nx_obs)
            comp_array2 = np.transpose(np.array( [list(np.array(xobs[:, obs_idx]))] * Nx_dat), [1,0,2])
            diffsqr = diffsqr + np.sum(np.abs(comp_array - comp_array2) /sigma[obs_idx], axis = 2)

        pyx = np.exp(-(diffsqr - np.transpose(np.matrix([[np.min(row) for row in diffsqr]] * Nx_dat))))
        pyxm = np.matrix([row / np.sum(row) for row in pyx.tolist()])
        px_new = np.array(np.sum(pyxm, 0))[0]
        px_new = px_new / sum(px_new)


        return px_new



    def initialize(self, Nx, init_snap, seed = 1, prob = []):

        #This initialize is different from the original in that it takes in the distribution
        np.random.seed(seed)

        if len(prob) == 0 :
            x0 = np.random.choice(init_snap, Nx)
        else:
            x0 = np.random.choice(init_snap, Nx, p = prob)

        return x0