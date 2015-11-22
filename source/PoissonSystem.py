import numpy as np
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
    dflt_param = [[[1., 0., 0.], [[0.], [0.]]],
                  [[0., 1., 0.], [[0.], [1.]]],
                  [[0., -2., 1], [[1.], [2.]]],
                  [[-1., 0., 0.], [[0.], [1.]]],
                  [[0., -1., 0.], [[1.], [1.]]]]

    dflt_theta = np.array([25., 1000., 0.001, 0.1, 1.])

    dflt_sigma = np.array([0.5, 1, 1])

    def __init__(self, theta_rxn_vec_pairs=dflt_param, theta = dflt_theta, sigma = dflt_sigma):
        self.param = theta_rxn_vec_pairs
        self.numrxn = len(self.param)
        self.sigma = sigma

        pre_rxnmatrix = np.matrix(np.zeros([len(self.param), len(self.param[0][0])]))
        #pre_theta = np.zeros(self.numrxn)
        reactant_v= []
        product_v =[]
        for k in range(0, self.numrxn):
            pre_rxnmatrix[k, :] = np.matrix(self.param[k][0])
            reactant_v = reactant_v + [self.param[k][1]]
            product_v = product_v + [self.param[k][0]]
            #pre_theta[k] = self.param[k][2]
        self.rxn_matrix = pre_rxnmatrix
        self.product = product_v
        self.theta = theta
        self.reactant = reactant_v

    def update_Gillespie(self, xnow, tnow):
        '''

        :param xdat: matrix([[state1], [state2], ...])
        '''
        nsample = len(xnow.tolist())
        rates = self.rate(np.asarray(xnow))
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
        numsamples = xinit.shape[0]

        if record:
            record_dat = np.zeros([maxjumps, numspecies])
            time_dat = np.zeros(maxjumps)
            record_dat[0, :] = xinit
            time_dat[0] = 0
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

    def update_Euler(self, xdat, tnow, deltat, record = False):

        #nsample = len(xnow.tolist())
        intensity = self.rate(np.asarray(xdat))
        rates = intensity * deltat

        #print rates
        #if np.sum(rates) > 1000: print np.sum(rates)

        rxn_cnt = np.random.poisson(rates)  # nsample x numrxn matrix
        deltax = rxn_cnt * self.rxn_matrix
        xnew = xdat + deltax
        tnew = tnow + deltat

        if record:
            return xnew, tnew, rxn_cnt, rates/self.theta
        else:
            return xnew, tnew

    def run_Euler(self, xinit, tend, deltat, tinit=0, seed = 2, record=False, recordtime=[]):

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


        param_record_dat = np.zeros([numsamples, self.numrxn])
        internal_integral = np.zeros([numsamples, self.numrxn])

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
            xnow, tnow, record_rxn, rates_paramless = self.update_Euler(xnow, tnow, deltat_now, record = True)
            param_record_dat = param_record_dat + record_rxn
            internal_integral = internal_integral + rates_paramless


        if record:
            return xnow, recordtime, record_dat
        else:
            return xnow, param_record_dat, internal_integral

    def rate(self, xdat):
        '''

        :param xdat:  matrix([[state1], [state2], ... ])
        :return:
        '''
        xdat = np.asarray(xdat)
        nsample = len(xdat.tolist())
        rate = np.zeros([nsample, self.numrxn])
        for k in range(0, self.numrxn):

            rate[:, k] = np.squeeze(self.theta[k]*np.power(xdat[:, self.reactant[k][0]], self.reactant[k][1]))

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


