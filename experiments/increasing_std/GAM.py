
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import random
import os
from numpy.random import multivariate_normal as randN
from global_data import *
import warnings

warnings.filterwarnings('ignore')

np.random.seed(1234)


def update_global_AM(optim_acc, t, acc_t, x, eps, lam, mu, sigma):
    gamma = 1 / (1 + t)**eps
    new_lam = np.exp(gamma*(acc_t - optim_acc)) * lam
    tmp = np.array(x) - mu
    new_mu = mu + gamma*tmp
    new_sigma = sigma + gamma * (np.outer(tmp, tmp) - sigma)
    return new_lam, new_mu, new_sigma


def Global_AM(likelihood_computer, prior, x_init, iterations, burn_in,
              data, acceptance_rule, eps=0.75, optim_acc=0.234):
    x = x_init
    d = len(x_init)
    samples = []
    accepted = []
    # initialize lambda to be the optimal scaling for SRWN
    lam = 2.38**2/d
    # initialize covariance sigma to be a small diagonal matrix
    sigma = np.identity(d)
    # mu is initialize to be a small value to avoid underflow
    mu = 1e-4 * np.ones(d)
    samples_index_array = []
    period_after_burn_in = 0
    accept_acc = 0
    for i in range(iterations):
        period_start_after_burnin = time.time()
        x_new = randN(x, lam * sigma)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        # MH step
        sig, acc_ratio = acceptance_rule(
            x_lik + np.log(prior(x)), x_new_lik+np.log(prior(x_new)))
        if sig:
            x = x_new
        if i > 0 and i % 100000 == 0:
            with open(f"{LOG_DIR}{os.sep}GAM-{i}", 'w') as f:
                f.write('')
        if i >= burn_in:
            samples.append(x)
            if sig == 1:
                accept_acc += 1
        lam, mu, sigma = update_global_AM(
            optim_acc, i, acc_ratio, x, eps, lam, mu, sigma)
        # report all the accepted states as well
        accepted.append(x)

        if i > burn_in:
            period_after_burn_in += (time.time() - period_start_after_burnin)

        if i > burn_in and period_after_burn_in >= 10:
            period_after_burn_in = 0
            samples_index_array.append(len(samples))

    # average acceptance ratio
    avg_acc = accept_acc / (iterations - burn_in)
    return np.array(samples), avg_acc, np.array(accepted), np.array(samples_index_array)


samples_GAM, avg_acc, accepted_GAM, samples_index_array_RBAM = \
    Global_AM(log_posterior, prior, x_init, 2000000, 800000, [], acceptance)


np.save(f'{RESULTS_DIR}{os.sep}samples_GAM_{exp_name}.npy', samples_GAM)
np.save(f'{RESULTS_DIR}{os.sep}accepted_GAM_{exp_name}.npy', accepted_GAM)
np.save(f'{RESULTS_DIR}{os.sep}samples_index_array_GAM_{exp_name}.npy', samples_index_array_RBAM)
