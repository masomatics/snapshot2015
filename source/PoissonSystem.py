"""
PoissonSystem.py

"""



import numpy as np
from scipy.special import factorial
import util_Snap as util
infinima = np.power(10,-20.)

__author__ = 'markov'

class PoissonSystem:
    '''
    Default system is  \
    0 -> 0 + M   25
    M -> M + P   1000
    2P-> D       0.001
    M -> 0       0.1
    P -> 0       1

    parameter convention: [rxnvector, listof[reactant specie idx, order], rate]
    '''
    dflt_param = [[[1., 0., 0.], [[0], [0.]]],
                  [[0., 1., 0.], [[0], [1.]]],
                  [[0., -2., 1], [[1], [2.]]],
                  [[-1., 0., 0.], [[0], [1.]]],
                  [[0., -1., 0.], [[1], [1.]]]]

    dflt_theta = np.array([25., 1000., 0.001, 0.1, 1.])
    dflt_sigma = np.array([0.5, 1, 1])

    def __init__(self, kinetics=dflt_param, theta = dflt_theta, sigma = dflt_sigma):
        self.kinetics = kinetics
        self.numrxn = len(self.kinetics)
        self.sigma = sigma

        pre_rxnmatrix = np.matrix(np.zeros([len(self.kinetics), len(self.kinetics[0][0])]))
        #pre_theta = np.zeros(self.numrxn)
        reactant_v= []
        product_v =[]
        for k in range(0, self.numrxn):
            pre_rxnmatrix[k, :] = np.matrix(self.kinetics[k][0])
            reactant_v = reactant_v + [self.kinetics[k][1]]
            product_v = product_v + [self.kinetics[k][0]]
            #pre_theta[k] = self.kinetics[k][2]
        self.rxn_matrix = pre_rxnmatrix
        self.product = product_v
        self.theta = theta
        self.reactant = reactant_v
    """
    def __init__(self, rxn_matrix= dflt_rxnMat, theta=dflt_theta, reactant=dflt_reactant, sigma = dflt_sigma):
        self.rxn_matrix = rxn_matrix
        self.reactant = reactant
        self.sigma = sigma
    """

    def update_Gillespie(self, xnow, tnow):
        '''

        :param xdat: matrix([[state1], [state2], ...])
        '''
        nsample = len(xnow.tolist())
        rates = self.rate(np.asarray(xnow))*self.theta
        ratesum = np.sum(rates, axis=1)
        probability = rates/ratesum.reshape(nsample, 1)
        deltat = -np.array(np.log(np.random.uniform(0.0, 1.0, 1)))/ratesum.reshape(nsample, 1)
        choices = range(0, self.numrxn)
        rxn = [np.random.choice(choices, 1, p=probability[k]) for k in range(0, nsample)]
        xnew = xnow + np.squeeze(np.array(self.rxn_matrix[rxn, :]))
        tnew = deltat + tnow
        return xnew, tnew


    def run_Gillespie(self, xinit, tend, tinit= 0, record = False, maxjumps = 30000):
        '''
        This makes only one trajectory.
        :param xinit:  np.matrix([[state]])   1 sample ONLY!
        :param tend:
        :param tinit:
        :param record:
        :return:
        '''

        numerical_precision = 0.000000001
        xnow = xinit
        tnow = tinit
        numspecies = xinit.shape[1]

        if record:
            record_dat = np.zeros([maxjumps, numspecies])
            time_dat = np.zeros(maxjumps)

            record_dat[0, :] = xinit
            time_dat[0] = tinit
        jump_idx = 0
        while(tnow < tend and jump_idx < maxjumps):
            xnow, tnow = self.update_Gillespie(xnow, tnow)
            jump_idx +=1
            if record:
                #print [np.squeeze(np.asarray(xnow)[0]).tolist(), np.squeeze(tnow).tolist()]
                record_dat[jump_idx, :] = np.squeeze(np.asarray(xnow)[0]).tolist()
                time_dat[jump_idx] = tnow
        if record:
            return xnow, record_dat[range(0, jump_idx)], time_dat[range(0,jump_idx)]
        else:
            return xnow

    def update_Euler(self, xdat, tnow, deltat, mytheta = None, record = False, record_full_logp = False):

        if mytheta == None:
            mytheta = self.theta.transpose()


        #nsample = len(xnow.tolist())
        intensity = self.rate(np.asarray(xdat))
        fullintensity = np.multiply(intensity, mytheta)
        lambdas = fullintensity * deltat
        lambdas = np.maximum(lambdas, infinima)

        rxn_cnt = np.random.poisson(lambdas)  # nsample x numrxn matrix
        deltax = rxn_cnt * self.rxn_matrix
        xnew = xdat + deltax
        tnew = tnow + deltat
        xnew = np.maximum(xnew, 0)
        if record:
            if record_full_logp:
                logp = - lambdas + rxn_cnt*np.log(lambdas)  - util.log_factorial(rxn_cnt)
                logp = np.sum(logp, axis = 1)
                #print np.log(factorial(rxn_cnt))
                return xnew, tnew, logp
            else:
                return xnew, tnew, rxn_cnt, lambdas/self.theta
        else:
            return xnew, tnew

    def run_Euler(self, xinit, tend, deltat, tinit=0, seed = 2, record=False, record_full_logp = False , recordtime=[]):

        '''

        :param xinit:
        :param tend:
        :param deltat:
        :param tinit:
        :param record:
        :param recordtime:
        :return:
        '''

        np.random.seed(seed)

        numerical_precision = 0.000000001
        numspecies = xinit.shape[1]
        numsamples = xinit.shape[0]
        xnow = xinit
        tnow = tinit
        record_idx = 0
        if(len(recordtime) == 0):
            recordtime =  np.linspace(tinit, tend, np.int((tend-tinit)/deltat) + 1)
        nextmark = recordtime[record_idx]
        #record_dat = [None] * len(recordtime)
        if record:
            record_dat = np.zeros([recordtime.shape[0], numsamples, numspecies])


        rxn_record_dat = np.zeros([numsamples, self.numrxn])
        internal_integral = np.zeros([numsamples, self.numrxn])
        record_logp = np.zeros([1, numsamples])

        while tend >tinit and tnow < (tend + numerical_precision):
            if np.abs(nextmark - tnow) < numerical_precision and record == True:
                #record_dat[record_idx] = [recordtime[record_idx], xnow]
                record_dat[record_idx, :, :] = xnow
                if record_idx < len(recordtime) -1:
                    record_idx += 1
                    nextmark = recordtime[record_idx]
            deltat_now = deltat + 0
            if tnow + deltat > nextmark and tnow < nextmark:
                deltat_now = nextmark - tnow

            #This is horrible conditioning.
            if deltat_now + tnow < (tend + numerical_precision):
                if record_full_logp:
                    xnow, tnow, logpDelta =  self.update_Euler(xnow, tnow, deltat_now, record = True, record_full_logp = True)
                    record_logp = record_logp + logpDelta
                else:
                    xnow, tnow, record_rxn, rates_paramless = self.update_Euler(xnow, tnow, deltat_now, record = True)
                    rxn_record_dat = rxn_record_dat + record_rxn
                    internal_integral = internal_integral + rates_paramless
            else:
                tnow = deltat_now + tnow

        #THIS IS SUCH A BAD conditions!!!!!
        if record_full_logp:
            if record:
                return xnow, recordtime, record_dat, record_logp
            else:
                return xnow, rxn_record_dat, internal_integral, record_logp
        else:
            if record:
                return xnow, recordtime, record_dat
            else:
                return xnow, rxn_record_dat, internal_integral

    def rate(self, xdat):
        '''
        THIS IS parameter free rates!!!!
        :param xdat:  matrix([[state1], [state2], ... ])
        :return:
        '''
        xdat = np.asarray(xdat)
        nsample = len(xdat.tolist())
        rate = np.zeros([nsample, self.numrxn])
        for k in range(0, self.numrxn):

            #print self.reactant[k][0]
            #print xdat[:, self.reactant[k][0]]
            rate[:, k] = np.squeeze(np.prod(np.power(xdat[:, self.reactant[k][0]], self.reactant[k][1][0]), axis = 1))

        rate = np.maximum(rate, 0)
        return rate

    def make_snapshots(self, snaptimes, init_snaps, observed, euler= True, delta = 0.01, default = True, nx = 1000, seed = 2):
        '''

        :param snaptimes: times from which to sample
        :param init_snap: list of initial distribution
        :param observed:  observed dimensions
        :param euler:     if true, euler tauleap
        :param delta:     delta of euler tauleap
        :param default:   if true, create init_snaps automatically
        :param nx:        if default is true, the size of the init_snaps to be made automatically
        :return:   dict snapshots{time, [obsv_index, dataset]}
        '''
        snapshots = {}


        for index in range(0, len(snaptimes)):
            if default:
                xdatinit = np.array([init_snaps[0]]*nx)
            else:
                xdatinit = init_snaps[index]
            if euler:
                xdat, rec, integral = self.run_Euler(xdatinit, snaptimes[index], delta, seed = seed)
            else:
                print "Gillespie version is under construction."
                pass
            snapshots[snaptimes[index]] = [None]*2
            for obsv_index in range(0, len(observed)):
                snapshots[snaptimes[index]][obsv_index]= [observed[obsv_index],  xdat[:,observed[obsv_index]]]

        return snapshots


    def predict_tau_ahead(self, xdistr, bigDeltat, deltat, tau, xstimP, xstimB, stim_target_index, target_index) :
        """
        For now, let is just work with this specific case.
        """
        xnow = xdistr
        N, numspecies = xdistr.shape
        record_logp_tau = np.zeros([1,N])
        target_taupath = np.zeros([N, tau])
        for k in range(0, tau):
            #stimulus this round
            xstim = np.array([xstimP[:,k],xstimB[:,k]] )
            #apply the stimulus
            xnow[:, stim_target_index] = xnow[:, stim_target_index] +  np.transpose(xstim)
            #move the prediction forward in time
            xnow, rxn_record_dat, internal_integral, record_logp =  self.run_Euler(xnow, bigDeltat, deltat, tinit=0, seed = 2, \
            record=False, record_full_logp = True , recordtime=[])
            target_taupath[:, k] = np.transpose(np.array(xnow[:, target_index]))
            record_logp_tau = record_logp_tau + record_logp

        return xnow, target_taupath, record_logp_tau
