import numpy as np
import pandas as pd
import matplotlib as mpl
import time
import random
from numpy.random import multivariate_normal as randN
import os
from global_data import *


np.random.seed(1234)


def Vanilla_MH(likelihood_computer, prior, x_init, iterations, burn_in, data, acceptance_rule):
    x = x_init
    d = len(x_init)
    samples = []
    accepted = []
    # initialize covariance sigma to be identity matrix
    sigma = 2.38**2/d * np.identity(d)
    accept_acc = 0
    samples_index_array = []
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
            x = x_new
        if i > 0 and i % 100000 == 0:
            with open(f"{LOG_DIR}{os.sep}MH-{i}", 'w') as f:
                f.write('')
        if i >= burn_in:
            samples.append(x)
            if sig == 1:
                accept_acc += 1
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


samples_MH, avg_acc, accepted_MH, samples_index_array_MH = \
    Vanilla_MH(log_posterior, prior, x_init, 2000000, 800000, [], acceptance)


np.save(f'{RESULTS_DIR}{os.sep}samples_MH_{exp_name}.npy', samples_MH)
np.save(f'{RESULTS_DIR}{os.sep}accepted_MH_{exp_name}.npy', accepted_MH)
np.save(f'{RESULTS_DIR}{os.sep}samples_index_array_MH_{exp_name}.npy', samples_index_array_MH)
