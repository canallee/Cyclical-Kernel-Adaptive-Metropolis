
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def update_RB_AM(acc_t, prev_x, x_prime, gamma, mu, sigma):
    prev_x = np.array(prev_x)
    x_prime = np.array(x_prime)
    tmp_prev_x = prev_x - mu
    tmp_x_prime = x_prime - mu
    # updating mu
    new_mu = mu + gamma * (acc_t * tmp_x_prime + (1-acc_t) * tmp_prev_x)
    # updating sigma
    tmp_1 = acc_t * np.outer(tmp_x_prime, tmp_x_prime)
    tmp_2 = (1-acc_t) * np.outer(tmp_prev_x, tmp_prev_x)
    new_sigma = sigma + gamma * (tmp_1 + tmp_2 - sigma)
    return new_mu, new_sigma


def Rao_Blackwellised_AM(likelihood_computer, prior, x_init,
                         iterations, burn_in, data, acceptance_rule,
                         gamma=0.1):
    x = x_init
    d = len(x_init)
    samples = []
    accepted = []
    # initialize covariance sigma to be a small diagonal matrix
    sigma = np.identity(d)
    # mu is initialize to be a small value to avoid underflow
    mu = 1e-4 * np.ones(d)
    samples_index_array = []
    accept_acc = 0
    period_after_burn_in = 0
    for i in range(iterations):
        period_start_after_burnin = time.time()
        x_new = randN(x, sigma)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        # MH step
        sig, acc_ratio = acceptance_rule(
            x_lik + np.log(prior(x)), x_new_lik+np.log(prior(x_new)))
        if sig:
            x_accept = x_new
        else:
            x_accept = x  # next state is same as previous one
        if i >= burn_in:
            samples.append(x_accept)
            if sig == 1:
                accept_acc += 1
        mu, sigma = update_RB_AM(acc_ratio, x, x_new, gamma, mu, sigma)
        x = x_accept
        if i > 0 and i % 100000 == 0:
            with open(f"{LOG_DIR}{os.sep}RBAM-{i}", 'w') as f:
                f.write('')
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


samples_RBAM, avg_acc, accepted_RBAM, samples_index_array_RBAM = Rao_Blackwellised_AM(
    counter_posterior, prior, X_INIT, 4000000, 1600000, [], acceptance)


np.save(f'{RESULTS_DIR}{os.sep}samples_RBAM_counter_example.npy', samples_RBAM)
np.save(f'{RESULTS_DIR}{os.sep}accepted_RBAM_counter_example.npy', accepted_RBAM)
np.save(f'{RESULTS_DIR}{os.sep}samples_index_array_RBAM.npy', samples_index_array_RBAM)
