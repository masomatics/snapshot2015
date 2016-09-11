__author__ = 'markov'


import numpy as np
import sys
sys.path.append("../source")
sys.path.append("../source")
from datetime import datetime

import Discrete_Doucet_system as dd
import Numerical_test_Doucet as nt
import pickle
reload(dd)
reload(nt)

nxobs = 1000
nxtest = 20000
t_end = 20
this_repeat = 4
Nx = 1000
my_sigma = 0
#my_sigma = 0.01  #CHAMPION
myheat = 0.999
#alpha0 = 0
alpha0 =0  #CHAMPION

#testseed = 2
#Nx_pretrain = 10000
#iter_pretrain = 100

nxs = [1000, 1000, 1000, 1000, 1000] #CHAMPION SET
#times = [0, 10, 15, 20, 40] #CHAMPION SET

times = [0, 10, 15, 30, 40] # This set does not work when alpha = 0


#MAKE SNAPSHOTS
dsystem = dd.Discrete_Doucet_system()
snapshots= dsystem.make_snapshots(nxs, times, np.array([-1.5]))

'''
print "PRETRAINING SEQUENCE..."

theta_init = np.random.uniform(-1, 1 , 5)
pre_test_seq = nt.DM_test(alpha =0, theta_init= theta_init)
theta_init = pre_test_seq.pretrain(iter_pretrain, Nx_pretrain ,snapshots, theta_init)
print "...COMPLETE"

theta_init[1] = np.min([np.abs(theta_init[1]), 1])
theta_init[4] = 0.2

print theta_init
'''


#This is what can be obtained from the above pretraining sequence
#theta_init = np.array([-1.31730272,  0.01501897,  0.08615663,  0.08950387,  0.2       ])

theta_inits  = pickle.load( open( "../records/thetahistory_multiple_slices2016_911_200.p", "rb" ) )


#print theta_inits
#print theta_inits.shape
theta_variances = np.zeros(theta_inits.shape)

for theta_init_index in range(0, len(theta_inits)):


    theta_init =  theta_inits[theta_init_index]
    print theta_init
    test_seq = nt.DM_test(alpha =alpha0, theta_init= theta_init)


    theta_variance = test_seq.run_Estep_variance_test(Nx, snapshots, theta_init, write = False, heat = 0.99, mysigma = 0, myseed = 0, repeats = this_repeat)

    theta_variances[theta_init_index] = theta_variance

print theta_variances

d = datetime.now()
timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute)
variance_filename = "../records/variance_records_alpha" + str(alpha0) +  timenow + ".p"

pickle.dump(theta_variances, open(variance_filename, "wb"))
