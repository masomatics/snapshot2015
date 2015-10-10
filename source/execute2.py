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
alpha0 = 1
t_end = 20
n_iter = 1500
Nx = 1000

#nxs = [1000, 2000, 1000]
#times = [0, 8, 15]


#nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
#times = [0, 10, 15, 20, 40] #CHAMPION SET

nxs = [1000, 1000, 1000, 1000]
times = [0, 10, 20, 40]

dsystem = dd.Discrete_Doucet_system()
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))

#theta_init = np.array([0, 0, 0, 0.1, 0.2])
theta_init =np.random.normal(0,0.5,5)

#Champion Initial condition
#theta_init =np.array([-1.09639492,  0.17766198, -0.19591798,  0.57649007,  0.2       ])

theta_init[4] = 0.2

test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write=True)

