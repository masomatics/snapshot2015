import numpy as np
import  main
reload(main)
from timeit import default_timer as timer


theta_approx0 = np.array([-1.5, 0, 0, 0.1, 0.2])
#theta_approx0 = np.array( [ -2.06345867,  -0.40571014, -26.41343345 ,  6.47108246 ,  0.2       ])
nxobs = 1000
nxtest = 20000
alpha0 = 0.1
t_end = 40
n_iter = 2000

test_seq = main.DM_test(Nx_obs = nxobs, Nx_test = nxtest, alpha =alpha0, tend = t_end, theta_init= theta_approx0)


test_seq.run(n_iter,write = False)
