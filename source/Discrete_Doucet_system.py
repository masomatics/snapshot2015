import numpy as np

__author__ = 'markov'


class Discrete_Doucet_system:

    dflt_theta = np.array([0, 1./2., 25., 6., 0.2])
    dflt_sigma= np.array([np.sqrt(10), np.sqrt(5)])
    def __init__(self, theta = dflt_theta, sigma=dflt_sigma ):
        self.theta = theta
        self.sigma = sigma

    def update(self, xdat, pdat):
        """

        :param xdat: np array of current values
        :param pdat: np array of likelihoods now
        :return: np array of updated values
        """

        sys_noise =  np.random.normal(0., self.sigma[0], len(xdat))
        #obs_noise = np.random.normal(0., self.sigma[1], len(xdat))
        xdat_new =  np.sum(self.fval(xdat), 0 ) + sys_noise
        #ydat_new = xdat_new + obs_noise
        xlog_likelihood  = np.log(np.power(sys_noise,2.))
        pdat = pdat + xlog_likelihood
        #ylog_likelihood = np.log(np.power(obs_noise,2.))


        return np.array(xdat_new)[0], pdat

    def initialize(self, Nx, init):
        """

        :param Nx : integer indicating the number of samples
        :return: x0
        """
        init_noise = np.random.normal(0, self.sigma[1], Nx)
        x0 = init + init_noise
        p0 = np.log(np.power(init_noise,2.))
        return x0,p0

    def fval(self, xdat):
        """

        :rtype : matrix
        """
        fvalue = np.matrix([[self.theta[0]]*len(xdat), self.theta[1]*xdat, self.theta[2]* \
                            (xdat / (1. + np.power(xdat, 2)) ), self.theta[3]*np.cos(self.theta[4]*xdat)])
        return  fvalue

    def jacob(self, xdat, sine = True):
        """

        :param xdat:
        :param sine: boolean of whether I include since function
        :returns : np matrix del f del x
        """
        if sine:
            jacobian =  np.matrix([[1.]*len(xdat),  xdat, (xdat / (1. + np.power(xdat,2)) ), \
                     np.cos(self.theta[4]*xdat), -xdat * self.theta[3] *np.sin(self.theta[4]*xdat) ])
        else:
            jacobian =  np.matrix([[1.]*len(xdat),  xdat, (xdat / (1. + np.power(xdat,2)) ), \
                     np.cos(self.theta[4]*xdat)])
        return jacobian



class Simulate:

    dflt_tend = 40
    dflt_Nx = 1000
    dflt_Ny = 1000
    dflt_init = -1.5

    def __init__(self, init = dflt_init, T = dflt_tend):
        #self.Nx = Nx
        #self.Ny = Ny
        self.tend = T
        self.init = init

    def simulate(self, dsystem):
        """
        :param Nx: number of samples to simulate
        :param dsystem: the Discrete_Doucet_system
        :return:
        """

        assert isinstance(dsystem,Discrete_Doucet_system)
        xdat, pdat = dsystem.initialize(Nx, self.init)
        for t in range(0,self.tend):
            xdat, pdat = dsystem.update(xdat, pdat)

        return xdat, pdat

    def moment_history(self, dsystem,  powers, Nx):

        assert isinstance(dsystem,Discrete_Doucet_system)
        assert isinstance(Nx,int)
        xdat, pdat = dsystem.initialize(Nx, self.init)
        mmt_history = np.zeros([self.tend +1, len(powers)])

        for t in range(0, self.tend+1):
            mmt_history[t] = np.array([np.mean(np.power(xdat, exponent)) for exponent in powers])
            xdat, pdat = dsystem.update(xdat, pdat)

        return np.transpose(mmt_history)