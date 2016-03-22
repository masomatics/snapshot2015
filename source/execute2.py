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



#nxs = [1000, 2000, 1000]
#times = [0, 8, 15]

#Champion set 1
alpha0 = 1 # CHAMPION ALPHA
nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
times = [0, 10, 15, 20, 40] #CHAMPION SET

#MAKE SNAPSHOTS
dsystem = dd.Discrete_Doucet_system()
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))



theta_init =np.array([-1.09639492,  0.17766198, -0.19591798,  0.57649007,  0.2       ])#Champion Initial condition


#print "PRETRAINING SEQUENCE..."
#theta_init = np.random.uniform(-1, 1 , 5)
#pre_test_seq = nt.DM_test(alpha =0, theta_init= theta_init)
#theta_init = pre_test_seq.pretrain(100, 50000,snapshots, theta_init)
#print "...COMPLETE"
#alpha0  = 100

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


#times = [0, 10, 15, 20, 40]
#nxs = [1000, 1000, 1000, 1000]
#times = [0, 10, 20, 40]
#times = [0, 5, 10, 20, 40]

#theta_init = np.array([0, 0, 0, 0.1, 0.2])
#theta_init =np.random.normal(0,0.1,5)
#theta_init = np.array([-0.08779999,  0.04190758 , 0.07800429, -0.10113304,  0.2       ])
theta_init[1] = np.min([np.abs(theta_init[1]), 1])
#theta_init[3] = abs(theta_init[3])







theta_init[4] = 0.2

test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write=False)
