import numpy as np
import  main
reload(main)
import timeit


theta_approx0 = np.array([-1.5, 0, 0, 0.1, 0.2])
nxobs = 1000
nxtest = 20000
alpha0 = 0.1
t_end = 15


test_seq = main.DM_test(Nx_obs = nxobs, Nx_test = nxtest, alpha =alpha0, tend = t_end, theta_init= theta_approx0)


start = timeit.timeit()
test_seq.run(200)
end = timeit.timeit()
print end - start