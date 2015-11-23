__author__ = 'markov'
import numpy as np
from PoissonSystem import PoissonSystem as pp
from Numerical_test import Numerical_test
import sys
import pickle
from datetime import datetime

smallnumber = 1e-6

class Numerical_test_Poi(Numerical_test):

    def __init__(self, alpha, theta_init):
        Numerical_test.__init__(self,  alpha, theta_init)

    def _compute_new_theta(self, snapshots, Nx, delta, psystem_old, observed, init_snap, seed=2, alpha = 0):

        '''

        :param snapshots:
        :param Nx:
        :param delta:
        :param psystem_old:
        :param px:
        :param observed:
        :param seed:
        :param alpha:
        :return:
        '''

        theta_approx = psystem_old.theta
        numrxn = len(theta_approx)


        times = np.sort(np.array([key for key in snapshots]))
        numslice = len(times)

        snap_old = init_snap
        px0 = []
        pathdat_numerator = np.zeros([1, numrxn])
        pathdat_denominator = np.zeros([1, numrxn])

        for k in range(0, numslice-1):


            time_old = times[k]
            time_new = times[k+1]  # Get this from iteritem
            snap_new = snapshots[time_new]
            #print snap_new
            snap_old_resample = self.initialize(Nx, snap_old, seed=seed*numslice + k, prob=px0)


            seed_common = seed * numslice+k+1

            #NEW weight with PX IT  IS CRITICAL THAT the PX and xdat_new is generated from the same SEED
            xdat_test, rxns_test, integral_test= psystem_old.run_Euler(snap_old_resample, tend=time_new,\
                                     deltat=delta, tinit=time_old, seed=seed_common, record=False, recordtime=[])


            xdat_old, rxns_old, integral_old= psystem_old.run_Euler(snap_old_resample, tend=time_new,\
                                     deltat=delta, tinit=time_old, seed=seed*numslice+k+2, record=False, recordtime=[])

            pxnew = self.compare_snap(xdat_test, snap_new, psystem_old.sigma, observed=observed)


            pathdat_numerator = pathdat_numerator + np.dot(rxns_test.transpose(),pxnew) + alpha*np.mean(rxns_old.transpose(),axis = 1)
            pathdat_denominator =pathdat_denominator + np.dot(integral_test.transpose(),pxnew) + alpha*np.mean(integral_old.transpose(),axis = 1)

            px0 = pxnew
            snap_old = xdat_test

        #print 'Denominator'
        #print pathdat_denominator.shape, pathdat_denominator

        #print 'Numerator'
        #print pathdat_numerator.shape, pathdat_numerator

        thetanew = np.sum(pathdat_numerator, axis = 0) / np.sum(pathdat_denominator, axis = 0)

        thetanew[np.where(thetanew == 0)] = smallnumber
        thetanew[np.isnan(thetanew)] = smallnumber



        return np.squeeze(thetanew)

    def em_algorithm(self, n_iter, Nx, snapshots, delta, theta_init, observed, init_snap, system, write = False, myalpha = 0.5):

        #Recording the current time
        d = datetime.now()
        timenow = str(d.year) + "_" + str(d.month) + str(d.day) + "_" + str(d.hour) + str(d.minute)
        stdout_filename = "../records/transition_multiple_slices" + timenow + ".txt"
        history_filename = "../records/thetahistory_multiple_slices" +  timenow + ".p"

        #Dirichlet parameters
        alpha = myalpha

        kinetics0 = system.kinetics
        sigma0 = system.sigma

        #history of parameters to save
        theta_history = np.zeros([n_iter+1, theta_init.shape[0]])
        theta_history[0] = theta_init
        theta_approx = theta_init
        if write:
            sys.stdout = open(stdout_filename, 'w')


        times = np.sort(np.array([key for key in snapshots]))
        nxs = [snapshots[time][0][1].shape[0] for time in snapshots.keys()]
        print "This computation is now done with new compute_AandB that does not use the intermediate snapshots as the inits"
        print "snaptimes are " + str(times)
        print "initial theta is: ", self.theta_init
        print "observed dimensions are:", observed
        print "Nx test is:", str(Nx)
        print "Nx observed are:", str(nxs)
        print "Alpha is :", str(self.alpha)
        print "INITIAL theta set at  " , theta_approx


        #BEGIN THE MAIN LOOP

        heat = 0.99999
        for iter in range(0, n_iter):

            print "iteration " + str(iter) + ":", theta_approx

            psystem_old = pp(kinetics = kinetics0, theta=theta_approx, sigma= sigma0)

            theta_approx = self._compute_new_theta(snapshots, Nx, delta, psystem_old, observed, init_snap, seed=2)

            theta_history[iter] = theta_approx

            alpha = alpha * heat
            print "alpha:", alpha

        pickle.dump(theta_history, open(history_filename, "wb"))
        return theta_approx



'''
    #This method is to be removed.
    def compare_snap(self, xdat, snapshot, sigma, obsv = [np.array([0])]):

        Nx_obs = len(snapshot[0][1])
        Nx_dat = len(xdat)

        diffsqr = np.zeros([Nx_obs, Nx_dat])

        for obs_idx in range(0,len(obsv)):

            dim_interest = obsv[obs_idx]
            comp_array = np.array([list(np.array(xdat[:, dim_interest]))] * Nx_obs)

            comp_array2 = np.transpose(np.array([snapshot[obs_idx][1]] * Nx_dat), [1, 0, 2])

            diffsqr = diffsqr + np.sum(np.abs(comp_array - comp_array2) /sigma[dim_interest], axis = 2)

        pyx = np.exp(-(diffsqr - np.transpose(np.matrix([[np.min(row) for row in diffsqr]] * Nx_dat))))
        pyxm = pyx / np.sum(pyx, axis=1)
        px_new = np.array(np.sum(pyxm, axis=0))[0]
        px_new = px_new / sum(px_new)

        return px_new
'''
