__author__ = 'markov'


import numpy as np
import sys
sys.path.append("../source")
import Discrete_Doucet_system as dd
import Numerical_test_Doucet as nt
reload(dd)
reload(nt)

nxobs = 1000
nxtest = 20000
t_end = 20
n_iter = 500
Nx = 1000
my_sigma = 0.01
myheat = 0.999
alpha0 = 1
Nx_pretrain = 10000
iter_pretrain = 100

nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
times = [0, 10, 15, 20, 40] #CHAMPION SET

#MAKE SNAPSHOTS
dsystem = dd.Discrete_Doucet_system()
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))


print "PRETRAINING SEQUENCE..."

theta_init = np.random.uniform(-1, 1 , 5)
pre_test_seq = nt.DM_test(alpha =0, theta_init= theta_init)
theta_init = pre_test_seq.pretrain(iter_pretrain, Nx_pretrain ,snapshots, theta_init)
print "...COMPLETE"

theta_init[1] = np.min([np.abs(theta_init[1]), 1])
theta_init[4] = 0.2

test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


theta_approx_last = test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write=False, mysigma = my_sigma, heat = myheat)

print(str(theta_approx_last))
