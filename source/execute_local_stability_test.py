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


test_alpha  =5.0         #CHANGE THIS!
num_trials = 300          #CHANGE THIS!!
theta_approxes = np.zeros([num_trials, 5])


nxobs = 1000
nxtest = 20000
t_end = 20
n_iter = 1 # THIS HAS TO BE FIXED AT ONE for local stability test
Nx = 1000
my_sigma = 0
#my_sigma = 0.01  #CHAMPION
myheat = 0.999
#alpha0 = 0
alpha0 = test_alpha  #CHAMPION

Nx_pretrain = 10000
iter_pretrain = 100      #CHANGE THIS!!!

nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
#times = [0, 10, 15, 20, 40] #CHAMPION SET

times = [0, 10, 15, 30, 40] # This set does not work when alpha = 0

'''
This is what can be obtained from pretraining with just mean for 100 iteration when the parameter is np.array([0, 0.5, 25, 6, 0.2])
'''
#theta_init = np.array([-1.38266457,  0.03831401,  0.25079858,  0.18597698,  0.2   ])

'''
This is what can be obtained from 100 iteration of the algorithm with successful parameters.
the parameter is drawn from the record 'thetahistory_multiple_slices2016_915_2028.p'
'''
#theta_init = np.array([ -1.746396  ,   0.54001062,  12.0395347 ,   4.34990197,   0.2       ])

'''
This is what can be obtained from pretraining with just mean for 100 iteration when the parameter is np.array([0.5, 0.7, 18, 2, 0.2])
'''
#theta_init = np.array([0.72016245,  0.02013309,  0.15208636,  0.02847037,  0.2])

'''
This is what can be obtained from 100 iteration of the algorithm with successful parameters.
the parameter is drawn from the record
'terminal_thetas_alpha_experiment2017_120_1749alpha_2.0_stdout.txt'
And particularly the one that produced
np.array([  0.51885506,   0.70482005,  17.74052002 ,  2.03955516 ,  0.2       ])
'''
theta_init = np.array([  0.94791151,   0.50280226,  21.73608789,  -2.44260004,   0.2       ])

#MAKE SNAPSHOTS
dsystem = dd.Discrete_Doucet_system(theta=theta_init)
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))

#Numerical test object
test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


d = datetime.now()
timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute) + "alpha_" + str(test_alpha)
std_filename = "../records/terminal_thetas_alpha_experiment" +  timenow + "_stdout.txt"
sys.stdout= open(std_filename, 'w')

print('In this experiment, I am running multiple sequences of ONE STEP of the snapshot learning and saving the terminal parameters to a file. The purpose of this experiment is to check the stability of the gradient at each step of the algorithm.')


for run_seed in range(num_trials):
    #Fix seed
    testseed = run_seed

    #n_iter = 1. Just one step.
    theta_approx_last = test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write=False, mysigma = my_sigma, myseed = testseed, heat = myheat)
    print(str(theta_approx_last))
    theta_approxes[run_seed, :] =  theta_approx_last

print theta_approxes

d = datetime.now()
timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute) + "alpha_" + str(test_alpha)

dataset_filename = "../records/one_theta_update_experiment" +  timenow + ".p"


pickle.dump(theta_approxes, open(dataset_filename, "wb"))
