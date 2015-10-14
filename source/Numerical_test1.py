
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

        def __init__(self,  alpha, theta_init):

                self.alpha = alpha
                self.theta_init = theta_init

        def run(self, n_iter, nx_obs, nx_test, write=False):


            #Recording the current time
            d = datetime.now()
            timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute)
            stdout_filename = "../records/transition" + timenow + ".txt"
            history_filename = "../records/thetahistory" + timenow + ".p"


            #Terminal time, system, simulation
            T0 = self.tend
            dsystem = dd.Discrete_Doucet_system()
            simul  = dd.Simulate(T = T0)
            xobs,pobs = simul.simulate(dsystem, Nx=nx_obs)

            #Dirichlet parameters
            alpha = self.alpha
            theta_history = np.zeros([n_iter+1, len(self.theta_init)])
            theta_history[0] = self.theta_init
            theta_approx = self.theta_init
            if write:
                sys.stdout = open(stdout_filename, 'w')

            print "terminal time is " + str(self.tend)
            print "initial theta is: ", self.theta_init
            print "Nx test is:", str(nx_test)
            print "Nx observed is:", str(nx_obs)
            print "Alpha is :", str(self.alpha)

            #Run the simulation
            start_time = timer()
            for iter in range(1,n_iter+1):
                print "iteration " + str(iter) + ":" , theta_approx
                dsystem_old = dd.Discrete_Doucet_system(theta = theta_approx)
                xdat_test, pdat_test= simul.simulate(dsystem_old, Nx = nx_test,  seed = iter*2)
                px = nx_test * dsystem.compare(xdat_test, xobs)
                xdat_new, pdat_new, A_new, B_new = simul.simulate(dsystem_old, Nx = nx_test,  seed = iter*2, Px = px, stat= True)
                xdat_old, pdat_old, A_old, B_old = simul.simulate(dsystem_old, Nx = nx_test,  seed = iter*2+1, stat= True)
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



            #Dirichlet parameters
            alpha = self.alpha
            theta_history = np.zeros([n_iter+1, theta_init.shape[0]])
            theta_history[0] = self.theta_init
            theta_approx = self.theta_init
            if write:
                sys.stdout = open(stdout_filename, 'w')

            times = np.sort(np.array([key for key in snapshots]))
            nxs = [snap.shape[0] for time, snap in snapshots.iteritems()]
            print "snaptimes are " + str(times)
            print "initial theta is: ", self.theta_init
            print "Nx test is:", str(Nx)
            print "Nx observed are:", str(nxs)
            print "Alpha is :", str(self.alpha)

            #PRECENT THE COLD START

            # A_try_sum = np.matrix(np.zeros((len(theta_init)-1, len(theta_init)-1) ))
            # B_try_sum = np.zeros(len(theta_init)-1)
            # n_booster = 10
            # search_sigma = 1.
            # search_alpha = 20
            # Nx_try = 100
            # for itn in range(0, n_booster):
            #     theta_try = theta_init +np.random.normal(0.,search_sigma,5)
            #     theta_try[4] = 0.2
            #     theta_try = np.abs(theta_try)
            #     dsystem_try = dd.Discrete_Doucet_system(theta=theta_try)
            #     A_try, B_try = self.__compute_A_and_B(snapshots, Nx_try, search_alpha, dsystem_try, itn)
            #     A_try_sum = A_try_sum + A_try
            #     B_try_sum = B_try_sum + B_try
            #     print "BOOST iteration" + str(itn) + ":"
            # theta_approx = np.array(np.linalg.inv(A_try_sum) * np.matrix(B_try_sum).transpose())
            # theta_approx = np.array(theta_approx.transpose().tolist()[0] + [0.2])
            # theta_approx[1] = np.min([abs(theta_approx[1]), 0.999])


            print "INITIAL theta set at  " , theta_approx


            #BEGIN THE MAIN LOOP

            heat = 0.99
            for iter in range(0, n_iter):

                print "iteration " + str(iter) + ":", theta_approx

                dsystem_old = dd.Discrete_Doucet_system(theta=theta_approx)

                A_soln, B_soln = self.__compute_A_and_B(snapshots, Nx, alpha, dsystem_old, iter)

                theta_approx = np.array(np.linalg.inv(A_soln) * np.matrix(B_soln).transpose())
                theta_approx = np.array(theta_approx.transpose().tolist()[0] + [0.2])

                theta_history[iter] = theta_approx

                alpha = alpha * heat
                print "alpha:", alpha

            pickle.dump(theta_history, open(history_filename, "wb"))
            return theta_approx





        def __compute_A_and_B(self, snapshots, Nx, alpha, dsystem_old, seed):

            '''
            Go through all snapshots and returns the update variable A and B
            :param snapshots:  dictionary (time, snapshot)
            :param Nx:         Number of simulations to be conducted for the inference
            :param alpha:      prior strength parameter in DP
            :param dsystem_old:  system
            :param seed:        seed*numslice +k, seed*numslice+k+1, seed*numslice+k+2  will be used
            :return:   np.matrix A ,  np.array B
            '''



            theta_approx = dsystem_old.theta
            numparam = len(theta_approx)
            A_soln = np.matrix(np.zeros((len(theta_approx)-1, len(theta_approx)-1) ))
            B_soln = np.zeros(len(theta_approx)-1)

            times = np.sort(np.array([key for key in snapshots]))
            nxs = [snap.shape[0] for time, snap in snapshots.iteritems()]
            numslice = len(times)


            for k in range(0, numslice-1):

                time_old=times[k]
                time_new=times[k+1]  # Get this from iteritem

                #time_old, time_new = np.sort(np.random.choice(times,2, replace=False))

                #print time_old, time_new

                snap_old = snapshots[time_old]
                snap_new = snapshots[time_new]

                snap_old_resample, pdatX= dsystem_old.initialize2(Nx, snap_old, continuous=True, seed = seed*numslice +k)

                seed_common = seed*numslice+k+1
                xdat_test, pdatX= dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old,  seed=seed_common )
                px = Nx * dsystem_old.compare(xdat_test, snap_new)

                #NEW weight with PX IT  IS CRITICAL THAT the PX and xdat_new is generated from the same SEED
                xdat_new, pdat_new, A_new, B_new = dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old, seed=seed_common, Px=px, stat=True)
                #OLD weight. PX is gone.
                xdat_old, pdat_old, A_old, B_old = dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old, seed=seed*numslice+k+2,  stat=True)

                A_soln = A_soln + (A_new + alpha *A_old)
                B_soln = B_soln + (B_new + alpha *B_old)




            return A_soln, B_soln