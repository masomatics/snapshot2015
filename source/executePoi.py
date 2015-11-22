__author__ = 'markov'


import numpy as np
import sys
import PoissonSystem as ps
reload(ps)
import cProfile
import Numerical_test as Numt
import Numerical_test_Poi as NumtP



small_number = 0.1
n_iter = 2000
delta = 0.001
Nx = 1000
Ny = 1000
alpha = 0.5


psystem = ps.PoissonSystem()
initsnap = np.array([[0,0,0]] * Nx)
observed = np.array([[0,1],[0,2]])
snaptimes =np.array([0, 0.5, 1.] )
snapshots = psystem.make_snapshots(snaptimes, initsnap, observed,  delta = 0.001,  nx = Ny)



ntp= NumtP.Numerical_test_Poi(0.5, psystem.theta)

theta_init = np.array([1, small_number,small_number,small_number,small_number])
print theta_init
ntp.em_algorithm(n_iter, Nx, snapshots, delta, theta_init, observed, initsnap, write = False, myalpha = alpha)