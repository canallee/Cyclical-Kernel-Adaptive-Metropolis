import numpy as np
import os


def log_posterior(x, data):
    res = 0
    for i in range(1, 6):
        cov = np.identity(2) * (i ** 1/2)
        mu = (i - 1) * 2.5
        X_minus_mu = x - mu
        res += np.exp(-0.5 * (X_minus_mu @ np.linalg.inv(cov) @ X_minus_mu)) / \
            np.sqrt(np.log(2 * np.pi) * (np.linalg.det(cov)))
    return np.log(res)

def prior(params):
    return 1

# Defines whether to accept or reject the new sample


x_init = np.zeros(2)
exp_name = 'increasing_std'

# Defines whether to accept or reject the new sample


def acceptance(log_pi, log_pi_new, proposal_ratio=1):
    accept = np.random.uniform(0, 1)
    # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
    # less likely x_new are less likely to be accepted
    acc_ratio = np.exp(log_pi_new-log_pi)*proposal_ratio
    sig = 0
    if accept < acc_ratio:
        sig = 1
    if acc_ratio > 1:
        acc_ratio = 1
    return sig, acc_ratio


if os.path.exists('log') is False:
    os.makedirs('log')

if os.path.exists('results') is False:
    os.makedirs('results')

RESULTS_DIR = 'results'
LOG_DIR = 'log'
