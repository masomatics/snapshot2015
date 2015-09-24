import numpy as np
import  main
reload(main)

theta_approx0 = np.array([-1.5, 0, 0, 0.1, 0.2])
test_seq = main.DM_test(Nx_obs = 1000, Nx_test = 20000, alpha =0.1, tend = 4, theta_init= theta_approx0)
test_seq.run(200)