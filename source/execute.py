import numpy as np
import  main
reload(main)
from timeit import default_timer as timer


theta_approx0 = np.array([-1.5, 0, 0, 0.1, 0.2])
#theta_approx0 = np.array( [ -0.77172504,  -0.48468371, -25.73231738,   6.42606596,   0.2       ])
nxobs = 1500
nxtest = 15000
alpha0 = 0.01
t_end = 12
n_iter = 500

test_seq = main.DM_test(Nx_obs = nxobs, Nx_test = nxtest, alpha =alpha0, tend = t_end, theta_init= theta_approx0)


test_seq.run(n_iter,write = False)
