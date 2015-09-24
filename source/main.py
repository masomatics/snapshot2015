
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import sys
import pickle
sys.path.append("../source")
import Discrete_Doucet_system as dd
reload(dd)


class DM_test:

        def __init__(self, Nx_obs, Nx_test, alpha, tend, theta_init):
                self.nx_obs = Nx_obs
                self.nx_test = Nx_test
                self.alpha = alpha
                self.tend = tend
                self.theta_init = theta_init


        def run(self, n_iter):

            T0 = self.tend
            dsystem = dd.Discrete_Doucet_system()
            simul  = dd.Simulate(T = T0)
            xobs,pobs = simul.simulate(dsystem, Nx = self.nx_obs)

            alpha = self.alpha
            theta_history = np.zeros([n_iter+1, len(self.theta_init)])
            theta_history[0] = self.theta_init
            theta_approx = self.theta_init
            sys.stdout = open('transition.txt', 'w')
            for iter in range(1,n_iter+1):
                print "iteration " + str(iter) + ":" , theta_approx
                dsystem_old = dd.Discrete_Doucet_system(theta = theta_approx)
                xdat_test, pdat_test= simul.simulate(dsystem_old, Nx = self.nx_test,  seed = iter*2)
                px = self.nx_test * dsystem.compare(xdat_test, xobs)
                xdat_new, pdat_new, A_new, B_new = simul.simulate(dsystem_old, Nx = self.nx_test,  seed = iter*2, Px = px, stat= True)
                xdat_old, pdat_old, A_old, B_old = simul.simulate(dsystem_old, Nx = self.nx_test,  seed = iter*2+1, stat= True)
                theta_approx = np.array(np.linalg.inv(A_new + self.alpha*A_old) * np.matrix(B_new + self.alpha*B_old).transpose())
                theta_approx = np.array(theta_approx.transpose().tolist()[0] + [0.2])
                theta_history[iter] = theta_approx
            pickle.dump(theta_history, open("thetahistory.p", "wb"))