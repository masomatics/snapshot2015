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
Nx = 500
my_sigma = 0
myheat = 0.99
alpha0 = 1

nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
times = [0, 10, 15, 20, 40] #CHAMPION SET

#MAKE SNAPSHOTS
dsystem = dd.Discrete_Doucet_system()
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))

theta_init =np.array([-1.09639492,  0.17766198, -0.19591798,  0.57649007,  0.2       ])#Champion Initial condition
theta_init[1] = np.min([np.abs(theta_init[1]), 1])
theta_init[4] = 0.2

test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


theta_approx_last = test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write=False, mysigma = my_sigma, heat = myheat)

print(str(theta_approx_last))


##Champion Set 2 (not anymore with new compute_AandB)
#snaptimes are [ 0  5 15 20 40]
#initial theta is:
#theta_init = np.array(  [-0.09378553 , 0.12342881 , 0.23145231, -0.12408102 , 0.2       ])
#Nx test is: 500
#Nx observed are: [1000, 1000, 1000, 1000, 1000]
#Alpha is : 1.0
#INITIAL theta set at   [-0.09378553  0.12342881  0.23145231 -0.12408102  0.2       ]

##CHampion Set 3 (not anymore with new compute_AandB)
#snaptimes are [ 0  5 15 20 40]
#initial theta is:
#theta_init = np.array( [ 0.00381594,  0.01735446 ,-0.03249388 , 0.05159941 , 0.2       ])
#Nx test is: 500
#Nx observed are: [1000, 1000, 1000, 1000, 1000]
#Alpha is : 1.0
#INITIAL theta set at   [ 0.00381594  0.01735446 -0.03249388  0.05159941  0.2       ]
