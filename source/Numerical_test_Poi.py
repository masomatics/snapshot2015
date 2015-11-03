__author__ = 'markov'
import numpy as np
import Numerical_test


class Numerical_test_Poi(Numerical_test):

    def _compute_n_and_int_Euler(self, snapshots, Nx, delta, psystem_old, px, observed, seed=2, alpha = 0):

        theta_approx = psystem_old.theta
        numrxn = np.zeros((len(theta_approx)-1))
        numpxs = np.zeros((len(theta_approx)-1))


        times = np.sort(np.array([key for key in snapshots]))
        numslice = len(times)

        snap_old = snapshots[0]
        px0 = []

        for k in range(0, numslice-1):


            time_old=times[k]
            time_new=times[k+1]  # Get this from iteritem


            snap_new = snapshots[time_new]
            snap_old_resample= self.intialize(Nx, snap_old, continuous=False, seed=seed*numslice +k, prob= px0)


            seed_common = seed * numslice+k+1

            xdat_test, rxns_test, integral_test= psystem_old.run_Euler(snap_old_resample, tend = time_new,\
                                     deltat=delta, tinit = time_old,seed = seed_common, record=False, recordtime=[])

            px = compare(xdat_test, xobs, sigma, observ)



            return numrxn, numpxs



        return numrxns, integral


    def compare_snap(xdat, snapshot, sigma, observed = [np.array([0])]):

        Nx_obs = len(snapshot[0][1])
        Nx_dat = len(xdat)

        diffsqr = np.zeros([Nx_obs, Nx_dat])


        for obs_idx in range(0,len(observed)):

            dim_interest = observed[obs_idx]
            comp_array = np.array( [list(np.array(xdat[:, dim_interest]))] * Nx_obs)
            comp_array2 = np.transpose(np.array( [snapshot[obs_idx][1]] * Nx_dat), [1,0,2])
            diffsqr = diffsqr + np.sum(np.abs(comp_array - comp_array2) /sigma[dim_interest], axis = 2)

        pyx = np.exp(-(diffsqr - np.transpose(np.matrix([[np.min(row) for row in diffsqr]] * Nx_dat))))
        pyxm = pyx / np.sum(pyx, axis = 1)
        px_new = np.array(np.sum(pyxm, axis = 0))[0]
        px_new = px_new / sum(px_new)

        return px_new