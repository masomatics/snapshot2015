__author__ = 'markov'
import numpy as np
import sys
sys.path.append("../source")
import Discrete_Doucet_system as dd
import Numerical_test_Doucet as nt
from datetime import datetime
import pickle

reload(dd)
reload(nt)


test_alpha  =2.0         #CHANGE THIS!
num_trials = 10          #CHANGE THIS!!
theta_approxes = np.zeros([num_trials, 5])


nxobs = 1000
nxtest = 20000
t_end = 20
n_iter = 5
Nx = 1000
my_sigma = 0
#my_sigma = 0.01  #CHAMPION
myheat = 0.999
#alpha0 = 0
alpha0 = test_alpha  #CHAMPION

Nx_pretrain = 10000
iter_pretrain = 2       #CHANGE THIS!!!

nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
#times = [0, 10, 15, 20, 40] #CHAMPION SET

times = [0, 10, 15, 30, 40] # This set does not work when alpha = 0

theta_init = np.array([-1.38266457,  0.03831401,  0.25079858,  0.18597698,  0.2   ])


#MAKE SNAPSHOTS
dsystem = dd.Discrete_Doucet_system()
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))

#Numerical test object
test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


for run_seed in range(num_trials):
    #Fix seed
    testseed = run_seed

    #n_iter = 1. Just one step.
    n_iter =1
    theta_approx_last = test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write=False, mysigma = my_sigma, myseed = testseed, heat = myheat)
    print(str(theta_approx_last))
    theta_approxes[run_seed, :] =  theta_approx_last

print theta_approxes

d = datetime.now()
timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute) + "alpha_" + str(test_alpha)

dataset_filename = "../records/one_theta_update_experiment" +  timenow + ".p"


pickle.dump(theta_approxes, open(dataset_filename, "wb"))
