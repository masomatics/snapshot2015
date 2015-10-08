
import numpy as np
import scipy as sp
from datetime import datetime
from timeit import default_timer as timer
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

        def run(self, n_iter, write = False):


            #Recording the current time
            d = datetime.now()
            timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute)
            stdout_filename = "../records/transition" + timenow + ".txt"
            history_filename = "../records/thetahistory" +  timenow + ".p"


            #Terminal time, system, simulation
            T0 = self.tend
            dsystem = dd.Discrete_Doucet_system()
            simul  = dd.Simulate(T = T0)
            xobs,pobs = simul.simulate(dsystem, Nx = self.nx_obs)

            #Dirichlet parameters
            alpha = self.alpha
            theta_history = np.zeros([n_iter+1, len(self.theta_init)])
            theta_history[0] = self.theta_init
            theta_approx = self.theta_init
            if write:
                sys.stdout = open(stdout_filename, 'w')

            print "terminal time is " + str(self.tend)
            print "initial theta is: ", self.theta_init
            print "Nx test is:", str(self.nx_test)
            print "Nx observed is:", str(self.nx_obs)
            print "Alpha is :", str(self.alpha)

            #Run the simulation
            start_time = timer()
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
            end_time = timer()
            print "elapsed time : "+ str(end_time - start_time)
            pickle.dump(theta_history, open(history_filename, "wb"))



        def run_multiple_snaps(self, n_iter, Nx, snapshots, theta_init, write = False):

            #Recording the current time
            d = datetime.now()
            timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute)
            stdout_filename = "../records/transition_multiple_slices" + timenow + ".txt"
            history_filename = "../records/thetahistory_multiple_slices" +  timenow + ".p"

            #
            theta_approx = theta_init


            #Dirichlet parameters
            alpha = self.alpha
            theta_history = np.zeros([n_iter+1, theta_init.shape[0]])
            theta_history[0] = self.theta_init
            theta_approx = self.theta_init
            if write:
                sys.stdout = open(stdout_filename, 'w')

            print "terminal time is " + str(self.tend)
            print "initial theta is: ", self.theta_init
            print "Nx test is:", str(self.nx_test)
            print "Nx observed is:", str(self.nx_obs)
            print "Alpha is :", str(self.alpha)

            times = [key for key in snapshots]
            numslice = len(times)
            for iter in range(0, n_iter):

                print "iteration " + str(iter) + ":", theta_approx

                dsystem_old = dd.Discrete_Doucet_system(theta=theta_approx)

                A_soln = np.matrix(np.zeros((len(theta_approx)-1, len(theta_approx)-1) ))
                B_soln = np.zeros(len(theta_approx)-1)

                for k in range(0, numslice-1):

                    time_old=times[k]
                    time_new=times[k+1]  # Get this from iteritem

                    snap_old = snapshots[time_old]
                    snap_new = snapshots[time_new]

                    snap_old_resample, pdatX= dsystem_old.initialize2(Nx, snap_old, continuous = True, seed = iter*numslice +k)

                    seed_common = iter*numslice+k+1
                    xdat_test, pdatX= dsystem_old.simulate(Nx, snap_old_resample , tend = time_new - time_old,  seed = seed_common )
                    px = Nx * dsystem_old.compare(xdat_test, snap_new)

                    #NEW weight with PX IT  IS CRITICAL THAT the PX and xdat_new is generated from the same SEED
                    xdat_new, pdat_new, A_new, B_new = dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old, seed=seed_common, Px=px, stat=True)
                    #OLD weight. PX is gone.
                    xdat_old, pdat_old, A_old, B_old = dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old, seed=iter*numslice+k+2,  stat=True)

                    A_soln = A_soln + (A_new + alpha *A_old)
                    B_soln = B_soln + (B_new + alpha *B_old)

                theta_approx = np.array(np.linalg.inv(A_soln) * np.matrix(B_soln).transpose())
                theta_approx = np.array(theta_approx.transpose().tolist()[0] + [0.2])

                theta_history[iter] = theta_approx

            pickle.dump(theta_history, open(history_filename, "wb"))
            return theta_approx

