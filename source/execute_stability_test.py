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





test_alpha  =10.0         #CHANGE THIS!
num_trials = 10          #CHANGE THIS!!
theta_approxes = np.zeros([num_trials, 5])

nxobs = 1000
nxtest = 20000
t_end = 20
n_iter = 500
Nx = 1000
my_sigma = 0
#my_sigma = 0.01  #CHAMPION
myheat = 0.999
#alpha0 = 0
alpha0 = test_alpha  #CHAMPION
theta_test = np.array([0.5, 0.7, 18, 2, 0.2]) #CHANGE THIS!
#Default is np.array([0, 0.5, 25, 6, 0.2])


Nx_pretrain = 10000
iter_pretrain = 100       #CHANGE THIS!!!

nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
#times = [0, 10, 15, 20, 40] #CHAMPION SET

times = [0, 10, 20, 30, 40] # This set does not work when alpha = 0


d = datetime.now()
timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute) + "alpha_" + str(test_alpha)
std_filename = "../records/terminal_thetas_alpha_experiment" +  timenow + "_stdout.txt"
sys.stdout= open(std_filename, 'w')

print('In this experiment, I am running multiple sequences of snapshot learning and saving the terminal parameters to a file.')



#MAKE SNAPSHOTS
dsystem = dd.Discrete_Doucet_system(theta = theta_test)
print "the true parameter is: " +  np.str(dsystem.theta)
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))

print "PRETRAINING SEQUENCE..."

#theta_init = np.random.uniform(-1, 1 , 5)
#pre_test_seq = nt.DM_test(alpha =0, theta_init= theta_init)
#theta_init0 = pre_test_seq.pretrain(iter_pretrain, Nx_pretrain ,snapshots, theta_init)

'''
This is what can be obtained from pretraining for this specific set with 100 iteration for default parameter.
'''
#theta_init = np.array([-1.38266457,  0.03831401,  0.25079858,  0.18597698,  0.2   ])

'''
This is what can be obtained from pretraining for this specific set with 100 iteration for [0.5, 0.7, 18, 2, 0.2]
'''
theta_init0 = np.array([0.72016245,  0.02013309,  0.15208636,  0.02847037,  0.2])
print "...COMPLETE"


for run_seed in range(num_trials):

    testseed = run_seed

    theta_init = theta_init0
    theta_init[1] = np.min([np.abs(theta_init[1]), 1])
    theta_init[4] = 0.2

    test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


    theta_approx_last = test_seq.run_multiple_snaps(n_iter, Nx, snapshots, theta_init, write=False, mysigma = my_sigma, myseed = testseed, heat = myheat)


    d = datetime.now()
    timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute) + "alpha_" + str(test_alpha)

    print(str(theta_approx_last))
    theta_approxes[run_seed, :] =  theta_approx_last

    print str(run_seed) +  "th Round...COMPLETE"


print theta_approxes

dataset_filename = "../records/terminal_thetas_alpha_experiment" +  timenow + ".p"

pickle.dump(theta_approxes, open(dataset_filename, "wb"))
