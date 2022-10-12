
import numpy as np
import pandas as pd
import time
import random
import os
from numpy.random import multivariate_normal as randN
from global_data import *


np.random.seed(1234)

# Vanilla AM, Algorithm 2 in "a tutorial on adatpive mcmc"
# mu:       intermediate variable for adaptation
# sigma:    covariance matrix for SRWN, with proposal probability P_SRWN = N(0, 2.38**2/d)
# gamma:    stepsize
# Returns:  1) new mu, 2) new sigma


def update_Vanilla_AM(x, mu, sigma, gamma):
    tmp = np.array(x) - mu
    new_mu = mu + gamma * tmp
    new_sigma = sigma + gamma * (np.outer(tmp, tmp) - sigma)
    return new_mu, new_sigma


def Vanilla_AM(likelihood_computer, prior, x_init, iterations, burn_in,
               data, acceptance_rule, gamma=0.1):
    x = x_init
    d = len(x_init)
    samples = []
    accepted = []
    # initialize covariance sigma to be identity matrix
    sigma = np.identity(d)
    mu = 1e-4 * np.ones(d)
    accept_acc = 0
    samples_index_array = []
    period_after_burn_in = 0
    for i in range(iterations):
        period_start_after_burnin = time.time()
        cov = 2.38**2/d * sigma
        x_new = randN(x, cov)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        # MH step
        sig, acc_ratio = acceptance_rule(
            x_lik + np.log(prior(x)), x_new_lik+np.log(prior(x_new)))
        if sig:
            x = x_new
        if i > 0 and i % 100000 == 0:
            with open(f"{LOG_DIR}{os.sep}AM-{i}", 'w') as f:
                f.write('')
        if i >= burn_in:
            samples.append(x)
            if sig == 1:
                accept_acc += 1
        mu, sigma = update_Vanilla_AM(x, mu, sigma, gamma)
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


samples_AM, avg_acc, accepted_AM, samples_index_array_AM = Vanilla_AM(
    counter_posterior, prior, X_INIT, 5000000, 2000000, [], acceptance)


np.save(f'{RESULTS_DIR}{os.sep}samples_AM_counter_example.npy', samples_AM)
np.save(f'{RESULTS_DIR}{os.sep}accepted_AM_counter_example.npy', accepted_AM)
np.save(f'{RESULTS_DIR}{os.sep}samples_index_array_AM.npy',
        samples_index_array_AM)
