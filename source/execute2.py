__author__ = 'markov'


import numpy as np
import sys
sys.path.append("../source")
import Discrete_Doucet_system as dd
import Numerical_test1 as nt
reload(dd)
reload(nt)

nxobs = 1000
nxtest = 20000
alpha0 = 0.01
t_end = 20
n_iter = 10
Nx = 500



nxs = [1000, 2000, 1000]
times = [0, 8, 15]
dsystem = dd.Discrete_Doucet_system()
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))
theta_init = np.array([-1.5, 0, 0, 0.1, 0.2])


test_seq = nt.DM_test(alpha =alpha0, tend =t_end, theta_init= theta_init)


test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write = False)

