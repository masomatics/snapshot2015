'''
Functional version of variance_test_normal.py
'''


import sys
import numpy as np
import scipy as sp
import datetime
import util_Snap as util
import random

#http://qiita.com/TomokIshii/items/3a26ee4453f535a69e9e
import matplotlib as mpl
mpl.use('Agg')


import matplotlib.pyplot as plt
import datetime
import matplotlib.animation as manimation


def run_variance_test(dimdat, N = 50, M =100, numexp = 1000,  deltamu =10, numticks = 40, myattention_index = [], histbins = 20, datalocation = '../records'):

    mysig = 1.
    mysigma = np.array([mysig, mysig])
    np.random.seed(7)
    #loc1 = 0.5
    alphas = np.linspace(0,1., numticks +1)

    num_obs = len(myattention_index)
    if num_obs == 0:
        myattention_index = range(dimdat)


    addendum = "L2deltamu" + str(int(deltamu**2)) + "_"

    filestring = addendum +"variance_test" + str(len(myattention_index)) + "outOf" + str(dimdat) +"Dimen.mp4"
    arrayfilestringAlpha = addendum +"variance_testAlpha" + str(len(myattention_index)) + "outOf" + str(dimdat) +"Dimen.npy"
    arrayfilestringOne = addendum +"variance_testOne" + str(len(myattention_index)) + "outOf" + str(dimdat) +"Dimen.npy"
    arrayfilestring_alphachoice = addendum + "alphachoices.npy"



    mu1 = util.normalize_and_rescale(np.ones([dimdat,1]),deltamu)
    mu2 = np.zeros([dimdat,1])
    #print "true mu is:   " + str(np.transpose(mu1))
    #print "initial mu  is:" + str(np.transpose(mu2))

    #create obs datasets
    sig1half = np.random.normal(loc = 0, scale = 1, size = [dimdat,dimdat])
    sig2half = np.random.normal(loc = 0, scale = 1, size = [dimdat,dimdat])
    sig1 = np.dot(sig1half, np.transpose(sig1half))
    sig2 = np.dot(sig2half, np.transpose(sig2half))

    datobs = np.random.multivariate_normal(mean = np.transpose(mu1)[0], cov = sig1, size = N)


    resoln = 500



    done = 0
    Alpha2besaved = np.zeros([len(alphas), 2, histbins] )
    One2besaved = np.zeros([len(alphas), 2, histbins] )

    myfilename = util.make_filename(filestring, location =datalocation)
    myfileAlpha= util.make_filename(arrayfilestringAlpha, location =datalocation)
    myfileOne = util.make_filename(arrayfilestringOne, location =datalocation)
    myfile_alpha_choices = util.make_filename(arrayfilestring_alphachoice, location =datalocation)


    for alpha in alphas:
        plt.title( u'hist of $(d(Est - \hat E))$  alpha: ' + np.str(alpha),size='24')
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

        #print qmean

        mu_estimate_alpha_distances = util.distances(mu_estimate_alpha)
        mu_estimate_One_distances = util.distances(mu_estimate_1)

        pre_histalpha = plt.hist(mu_estimate_alpha_distances, alpha = 0, bins = histbins)
        pre_histOne = plt.hist(mu_estimate_One_distances, alpha = 0, bins = histbins)
        histalpha = util.convert_hist_to_scatter(pre_histalpha)
        histOne   = util.convert_hist_to_scatter(pre_histOne)

        Alpha2besaved[done, :, :] =  histalpha
        One2besaved[done, :, :] =  histOne


        if np.mod(done, 10) == 0:
            print "complete" + str(done)

        done = done +1


    np.save(myfile_alpha_choices, alphas)
    np.save(myfileAlpha  ,Alpha2besaved)
    np.save(myfileOne  ,One2besaved)

    util.convert_hists2mp4(myfilename, Alpha2besaved, One2besaved, alphas)
