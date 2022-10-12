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
    # initialize covariance sigma to be identity matrix
    sigma = 15 * np.identity(d)
    accept_acc = 0
    for i in range(iterations):
        x_new = randN(x, sigma)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        # MH step
        sig, acc_ratio = acceptance_rule(
            x_lik + np.log(prior(x)), x_new_lik+np.log(prior(x_new)))
        if sig:
            x = x_new
        if i > 0 and i % 100000 == 0:
            with open(f"{LOG_DIR}{os.sep}MH-true-{i}", 'w') as f:
                f.write('')
        if i >= burn_in:
            samples.append(x)

    return samples


samples_MH = \
    Vanilla_MH(log_posterior, prior, x_init, 3000000, 1200000, [], acceptance)


np.save(f'{RESULTS_DIR}{os.sep}samples_MH_true_{exp_name}.npy', np.array(samples_MH))
