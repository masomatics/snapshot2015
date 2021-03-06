__author__ = 'markov'


import numpy as np
import sys
import PoissonSystem as ps
reload(ps)
import cProfile
import Numerical_test_Poi as NumtP



small_number = 0.1
n_iter = 5
delta = 0.001
Nx = 1000
Ny = 1000
alpha = 0.2


psystem = ps.PoissonSystem()
initsnap = np.array([[0,0,0]] * Nx)
observed = np.array([[0,1],[0,2]])
snaptimes =np.array([0, 0.5, 1.] )
snapshots = psystem.make_snapshots(snaptimes, initsnap, observed,  delta = 0.001,  nx = Ny)



ntp= NumtP.Numerical_test_Poi(alpha, psystem.theta)

theta_init = np.array([1, small_number,small_number,small_number,small_number])
#theta_init = np.array([  5.64196265e+001  , 8.76033346e+002 ,  1.63965073e-003 ,  1.00000000e-006,   5.22243048e-311])
theta_init = psystem.theta


print theta_init
ntp.em_algorithm(n_iter, Nx, snapshots, delta, theta_init, observed, initsnap, psystem, write = False, myalpha = alpha)
