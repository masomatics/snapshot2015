import numpy as np
__author__ = 'markov'
from timeit import default_timer as timer


class Discrete_Doucet_system:
    dflt_theta = np.array([0, 1. / 2., 25., 6., 0.2])
    dflt_sigma = np.array([np.sqrt(10), np.sqrt(5)])

    def __init__(self, theta=dflt_theta, sigma=dflt_sigma):
        self.theta = theta
        self.sigma = sigma

    def update(self, xdat, pdat):
        """
        Update the xdat by xdat + f(xdat).  Also reports the log of the likelihood

        :param xdat: np array of current values
        :param pdat: np array of likelihoods now
        :return: np array of updated values
        """

        sys_noise = np.random.normal(0., self.sigma[0], len(xdat))
        # obs_noise = np.random.normal(0., self.sigma[1], len(xdat))
        xdat_new = np.sum(self.fval(xdat), 0) + sys_noise
        # ydat_new = xdat_new + obs_noise
        xlog_likelihood = np.log(np.power(sys_noise, 2.))
        pdat = pdat + xlog_likelihood
        # ylog_likelihood = np.log(np.power(obs_noise,2.))


        return np.array(xdat_new)[0], pdat

    def initialize(self, Nx, init, seed =1):
        """
        Initizalizes the distribution at normal centered at init

        :param Nx : integer indicating the number of samples
        :return: x0
        """
        np.random.seed(seed)
        init_noise = np.random.normal(0, self.sigma[1], Nx)
        x0 = init + init_noise
        p0 = np.log(np.power(init_noise, 2.))
        return x0, p0

    def initialize2(self, Nx, init_snap, seed = 1, continuous= False):

        #This initialize is different from the original in that it takes in the distribution
        np.random.seed(seed)

        x0 = np.random.choice(init_snap, Nx)
        p0 = np.zeros([1, Nx])

        if continuous:
            noise = np.random.normal(0., self.sigma[1], Nx)

        x0 = x0 + noise

        return x0, p0


    def fval(self, xdat):
        """
        Computes f(xdat), the update value

        :returns: f(xdat)
        """
        fvalue = np.matrix([[self.theta[0]] * len(xdat), self.theta[1] * xdat, self.theta[2] * \
                            (xdat / (1. + np.power(xdat, 2))), self.theta[3] * np.cos(self.theta[4] * xdat)])
        return fvalue

    def jacob(self, xdat, sine=False):
        """
        Computes the Jacobian del f del x for the system

        :param xdat:
        :param sine: boolean of whether I include since function
        :returns : np matrix del f del x
        """
        if sine:
            jacobian = np.matrix([[1.] * len(xdat), xdat, (xdat / (1. + np.power(xdat, 2))), \
                                  np.cos(self.theta[4] * xdat), -xdat * self.theta[3] * np.sin(self.theta[4] * xdat)])
        else:
            jacobian = np.matrix([[1.] * len(xdat), xdat, (xdat / (1. + np.power(xdat, 2))), \
                                  np.cos(self.theta[4] * xdat)])
        return jacobian

    def compare(self, xdat, xobs):

        Nx_obs = len(xobs)
        Nx_dat = len(xdat)
        xdatmat = np.matrix([list(xdat)] * Nx_obs)
        xobsmat = np.matrix([list(xobs)] * Nx_dat)
        diff = np.abs(xdatmat - np.transpose(xobsmat))
        diffsqr = np.power(diff, 2)

        # THIS is being done to increase the precision of the exponentiation
        pyx = np.exp(-(diffsqr - np.transpose(np.matrix([[np.min(row) for row in diffsqr]] * Nx_dat))))
        pyxm = np.matrix([row / np.sum(row) for row in pyx.tolist()])
        px_new = np.array(np.sum(pyxm, 0))[0]
        px_new = px_new / sum(px_new)

        return px_new

    def simulate(self, Nx, init_xdat, tend, seed =2, Px=np.array([]), stat = False):
        """
        This version of the fxn simulate allows arbitrary distributional input.
        For the generation of mock-obs data, call this function multiple times.

        :param stat: Boolean. True if we want to return A and B
        :param Nx: number of samples to simulate
        :return:
        matrix A, B  (refer to the equation in the article)
        ndarray pdat(likelihood), xdat(terminal value)
        """
        np.random.seed(seed)

        xdat = init_xdat
        pdat = np.zeros([1, Nx])


        if len(Px) != Nx:
            Px = np.array([1.] * Nx)

        if stat:
            R = len(self.theta) - 1
            A = np.matrix(np.zeros([R, R]))
            B = np.zeros([1, R])
            for t in range(0, tend):
                fhi = self.jacob(xdat)
                A = A + np.matrix(np.array(self.jacob(xdat)) * Px) * np.transpose(self.jacob(xdat))
                xdat, pdat = self.update(xdat, pdat)
                B = B + (np.array(fhi) * np.array(xdat) * Px).sum(axis=1)

            return xdat, pdat, A, B

        else:
            for t in range(0, tend):
                xdat, pdat = self.update(xdat, pdat)
            return xdat, pdat

    def make_snapshots(self, nxs, times, init_snap):
        """
        :param nxs:  lists of the size of observed datapoints.
        :param times:   lists of times at which the snapshots were taken
        :param init_snap:  The initial snapshot. Default is Gauss around -1.5
        :return:  Tuple Snapshots. THe sets of  (time,  nx)
        """
        snapshots={}
        for k in range(0, len(times)):
            xdat0, pdat0 = self.initialize2(nxs[k], init_snap, continuous=True)
            snapshots[times[k]], pdatX = self.simulate(nxs[k], xdat0, tend=times[k])
        return snapshots

#This class must be refactored.
class Simulate:
    dflt_tend = 40
    dflt_init = -1.5

    def __init__(self, init=dflt_init, T=dflt_tend):
        # self.Nx = Nx
        # self.Ny = Ny
        self.tend = T
        self.init = init

    def simulate(self, dsystem, Nx, seed=1, Px=np.array([]), stat=False):
        """
        :param stat: Boolean. True if we want to return A and B
        :param Nx: number of samples to simulate
        :param dsystem: the Discrete_Doucet_system
        :return:
        """
        #assert isinstance(dsystem, Discrete_Doucet_system)
        np.random.seed(seed)
        xdat, pdat = dsystem.initialize(Nx, self.init)

        if len(Px) != Nx:
            Px = np.array([1.] * Nx)

        np.random.seed(seed +1)

        if stat:
            R = len(dsystem.theta) - 1
            A = np.matrix(np.zeros([R, R]))
            B = np.zeros([1, R])
            for t in range(0, self.tend):
                fhi = dsystem.jacob(xdat)
                A = A + np.matrix(np.array(dsystem.jacob(xdat)) * Px) * np.transpose(dsystem.jacob(xdat))
                xdat, pdat = dsystem.update(xdat, pdat)
                B = B + (np.array(fhi) * np.array(xdat) * Px).sum(axis=1)

            return xdat, pdat, A, B

        else:
            for t in range(0, self.tend):
                xdat, pdat = dsystem.update(xdat, pdat)
            return xdat, pdat

    def moment_history(self, dsystem, powers, Nx):

        #assert isinstance(dsystem, Discrete_Doucet_system)
        #assert isinstance(Nx, int)
        xdat, pdat = dsystem.initialize(Nx, self.init)
        mmt_history = np.zeros([self.tend + 1, len(powers)])

        for t in range(0, self.tend + 1):
            mmt_history[t] = np.array([np.mean(np.power(xdat, exponent)) for exponent in powers])
            xdat, pdat = dsystem.update(xdat, pdat)

        return np.transpose(mmt_history)
