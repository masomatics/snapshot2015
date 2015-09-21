__author__ = 'markov'


import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import sys
sys.path.append("/Users/markov/Dropbox/Research/OnGoing_Research/Snapshot/Python_experiment/source")
import Discrete_Doucet_system
reload(Discrete_Doucet_system)


dsystem = Discrete_Doucet_system.Discrete_Doucet_system()
dsystem.theta, dsystem.sigma


init = -1.5
Nx = 3000

x0 = np.array([init]*Nx )
x0 = np.random.normal(init,  np.sqrt(5) , Nx)


T = 12
xnow = x0
pnow = np.power(x0 - init,2)
xmean = np.array([0.]*(T+1))
xmean[0] = np.mean(x0)

for t  in range(1,T+1,1):
    xnow, pnow = dsystem.update(xnow, pnow)
    xmean[t] = np.mean(xnow)

simul  = Discrete_Doucet_system.Simulate(Nx = 3000, T = 12)
powers = [1,2,3]
mmts = simul.moment_history(dsystem, powers)
mmts.shape

plt.subplot(1,2,0)
xplot= plt.hist(xnow,bins=50, normed=True)
