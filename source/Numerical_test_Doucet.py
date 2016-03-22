
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

    '''
    This is a numerical test package Dirichlet based inference for Discrete Doucet System.

    Parameters:
    :alpha: Heat parameter for the Dirichlet prior
    :theta_init: initial theta

    Functionalities:
    :run: REMOVED
    :run_multiple_snaps:  Run the inference
    :__compute_A_and_B:   Approximate the parameters
    :pretrain:  Particle smoother based EM on the mean of the path.
    '''

    def __init__(self,  alpha, theta_init):

            self.alpha = alpha
            self.theta_init = theta_init

    def run_multiple_snaps(self, n_iter, Nx, snapshots, theta_init, write = False):
        '''
        runs numerical experiment with Dirichlet Process based inference method.
        :n_iter:  number of iterations
        :Nx:   number of simulated points
        :snapshtos: observed snapshots.
        :theta_init: initial theta.
        '''

        #Recording the current time
        d = datetime.now()
        timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute)
        stdout_filename = "../records/transition_multiple_slices" + timenow + ".txt"
        history_filename = "../records/thetahistory_multiple_slices" +  timenow + ".p"

        #Dirichlet parameters
        alpha = self.alpha

        #history of parameters to save
        theta_history = np.zeros([n_iter+1, theta_init.shape[0]])
        theta_history[0] = self.theta_init
        theta_approx = self.theta_init
        if write:
            sys.stdout = open(stdout_filename, 'w')

        times = np.sort(np.array([key for key in snapshots]))
        nxs = [snap.shape[0] for time, snap in snapshots.iteritems()]
        print "This computation is now done with new compute_AandB that does not use the intermediate snapshots as the inits"
        print "snaptimes are " + str(times)
        print "initial theta is: ", self.theta_init
        print "Nx test is:", str(Nx)
        print "Nx observed are:", str(nxs)
        print "Alpha is :", str(self.alpha)
        print "INITIAL theta set at  " , theta_approx


        #BEGIN THE MAIN LOOP

        heat = 0.99
        for iter in range(0, n_iter):

            print "iteration " + str(iter) + ":" + str(theta_approx)

            dsystem_old = dd.Discrete_Doucet_system(theta=theta_approx)

            A_soln, B_soln = self.__compute_A_and_B(snapshots, Nx, alpha, dsystem_old, iter)

            theta_approx = np.array(np.linalg.inv(A_soln) * np.matrix(B_soln).transpose())

            theta_approx = np.array(theta_approx.transpose().tolist()[0] + [0.2])

            theta_history[iter] = theta_approx

            alpha = alpha * heat
            print "alpha:", alpha

        pickle.dump(theta_history, open(history_filename, "wb"))

        lastfew = np.int(np.ceil(np.double(n_iter)/10.))
        range_lastfew = range(n_iter-lastfew, n_iter)
        theta_approx = np.mean(theta_history[range_lastfew], axis = 0)         
        return theta_approx


    def __compute_A_and_B(self, snapshots, Nx, alpha, dsystem_old, seed):

        '''
        This function Go through all snapshots and returns the update variable A and B (refer to notes for the notation)
        *REMINDER* THE LAST PARAMETER IS EXCLUDED FROM THE SUBJECTS OF  INFERENCE!!!!
        :param snapshots:  dictionary (time, snapshot)
        :param Nx:         Number of simulations to be conducted for the inference
        :param alpha:      prior strength parameter in DP
        :param dsystem_old:  system
        :param seed:        seed*numslice +k, seed*numslice+k+1, seed*numslice+k+2  will be used
        :return:   np.matrix A ,  np.array B
        '''


        theta_approx = dsystem_old.theta
        numparam = len(theta_approx)

        #THIS REMOVES THE LAST  PARAMETER FROM THE INFERENCE TARGET
        A_soln = np.matrix(np.zeros((len(theta_approx)-1, len(theta_approx)-1) ))
        B_soln = np.zeros(len(theta_approx)-1)

        times = np.sort(np.array([key for key in snapshots]))
        nxs = [snap.shape[0] for time, snap in snapshots.iteritems()]
        numslice = len(times)


        snap_old = snapshots[0]
        px0 = []

        for k in range(0, numslice-1):

            time_old=times[k]
            time_new=times[k+1]  # Get this from iteritem

            snap_new = snapshots[time_new]
            snap_old_resample, pdatX= dsystem_old.initialize2(Nx, snap_old, continuous=False, seed=seed*numslice +k, prob= px0)

            seed_common = seed * numslice+k+1
            xdat_test, pdatX= dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old, seed=seed_common)
            #xdat_test serves as new snap_old
            snap_old = xdat_test
            px0 = dsystem_old.compare(xdat_test, snap_new)
            px = Nx * px0

            #NEW weight with PX IT  IS CRITICAL THAT the PX and xdat_new is generated from the same SEED
            xdat_new, pdat_new, A_new, B_new = dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old, seed=seed_common, Px=px, stat=True)
            #OLD weight. PX is gone.
            xdat_old, pdat_old, A_old, B_old = dsystem_old.simulate(Nx, snap_old_resample, tend=time_new - time_old, seed=seed*numslice+k+2,  stat=True)

            A_soln = A_soln + (A_new + alpha *A_old)
            B_soln = B_soln + (B_new + alpha *B_old)




        return A_soln, B_soln


    def pretrain(self, n_iter,Nx,snapshots, theta_init):

        '''
        This method runs brute-force particle smoother based EM using the mean values of snapshots in hope to provide better initial theta for the main part of the method. This can be done with __compute_A_and_B with alpha =0 and snapshots being equal to the single point.
        inputs:
        :n_iter: number of EM round.
        :snapshots: snapshots to be used for the EM.
        :theta_init: theta_init for the EM.
        '''
        mean_snapshots = self.take_mean(snapshots)

        pretrain_theta_approx = self.run_multiple_snaps(n_iter, Nx, mean_snapshots, theta_init, write = False)
        return pretrain_theta_approx

    def take_mean(self, snapshots):
        '''
        This function takes in snapshots and take mean at every timepoint.
        inputs:
        :snapshots: observed Snapshots

        returns:
        :mean_snaps:
        '''

        numslice = len(snapshots.keys())
        mean_snaps = {}
        for key in snapshots.keys():
            mean_snaps[key] = np.array([np.mean(snapshots[key])])
        return mean_snaps
