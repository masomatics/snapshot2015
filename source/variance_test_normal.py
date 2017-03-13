import sys
import numpy as np
import scipy as sp
import datetime
import util_Snap as util
import random
import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as manimation


dimdat = 300
N = 100   #obs
M = 1000       #sim
numexp = 200

#N = 4   #obs
#M = 40       #sim

myobserved = np.array([0])
mysig = 1.
mysigma = np.array([mysig, mysig])
np.random.seed(6)
loc1 = 0
loc2 = 0.
alphas = np.linspace(0,1, 21)
graphlim =2
datelocation = "../records"


#np.random.seed(10)
mu1 = np.random.normal(loc= loc1, scale = 1, size = [dimdat,1])
mu2 = np.random.normal(loc= loc2, scale = 1, size = [dimdat,1])
#print "true mu is:   " + str(np.transpose(mu1))
#print "initial mu  is:" + str(np.transpose(mu2))

#create obs datasets
sig1half = np.random.normal(loc = 0, scale = 1, size = [dimdat,dimdat])
sig2half = np.random.normal(loc = 0, scale = 1, size = [dimdat,dimdat])
sig1 = np.dot(sig1half, np.transpose(sig1half))
sig2 = np.dot(sig2half, np.transpose(sig2half))

datobs = np.random.multivariate_normal(mean = np.transpose(mu1)[0], cov = sig1, size = N)


resoln = 500
myattention_index = [0]



FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=20, metadata=metadata)
fig = plt.figure()
#Alpha = plt.scatter([], [], color = 'red')
#One = plt.scatter([], [], color = 'blue')
Alpha, = plt.plot([], [], 'r-o')
One,  = plt.plot([], [], 'b-o')
#plt.xlim([0.0,2])

plt.ylim([0,numexp/graphlim])
done = 0

filestring = "variance_test" + str(len(myattention_index)) + "outOf" + str(dimdat) +"Dimen.mp4"
myfilename = util.make_filename(filestring, location ='../records')

with writer.saving(fig, myfilename, resoln):

    for alpha in alphas:
        plt.title( u'alpha: ' + np.str(alpha),size='24')
        mu_estimate_1     =  np.zeros([dimdat, numexp])
        mu_estimate_alpha = np.zeros([dimdat, numexp])

        for k in range(numexp):
            datsim = np.random.multivariate_normal(mean = np.transpose(mu2)[0], cov = sig2, size = M)
            diffsqr = util.compare(datsim, datobs, attention_index = myattention_index )
            pyx = util.create_pyx(diffsqr)
            qmean = np.dot(pyx, datsim)
            mu_estimate_1[:, k] = qmean

            alpha_pyx  = alpha * pyx +  (1-alpha) * 1./ M
            alphamean = np.dot(alpha_pyx, datsim)

            mu_estimate_alpha[:, k] = alphamean

        writer.grab_frame()
        #print qmean

        mu_estimate_alpha_distances = util.distances(mu_estimate_alpha)
        mu_estimate_One_distances = util.distances(mu_estimate_1)

        pre_histalpha = plt.hist(mu_estimate_alpha_distances, alpha = 0)
        pre_histOne = plt.hist(mu_estimate_One_distances, alpha = 0)
        histalpha = util.convert_hist_to_scatter(pre_histalpha)
        histOne   = util.convert_hist_to_scatter(pre_histOne)

        #model = TSNE(n_components=2, random_state=0)
        #twoD_estimate_alpha = model.fit_transform(np.transpose(mu_estimate_alpha) )
        #twoD_estimate_One   = model.fit_transform(np.transpose(mu_estimate_1) )


        if np.mod(done, 10) == 0:
            print "complete" + str(done)

        #l, = plt.plot([], [], color = 'red')
        #Alpha.set_offsets(histalpha )
        #One.set_offsets(histOne)
        Alpha.set_data(histalpha[1,:], histalpha[0,:])
        One.set_data(histOne[1,:], histOne[0,:])

        done = done +1
        #plt.scatter(mu_estimate_alpha[0,:], mu_estimate_alpha[1,:], color = 'blue')
        #plt.scatter(mu_estimate_1[0,:], mu_estimate_1[1,:], color = 'red')
