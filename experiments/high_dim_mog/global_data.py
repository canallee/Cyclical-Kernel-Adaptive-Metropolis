import numpy as np
import os
import scipy.special


def gen_fast_ndim_mog_posterior(side_len, cov, dim=2, spread=3):
    # Always centered at (0,0)
    # Only odd side_len supported
    diameter = cov[0, 0]*spread

    def log_posterior(x, data):
        center = np.floor(x/diameter)*diameter
        gridsize = 1

        temp = np.ones((gridsize,)*dim, dtype=np.float32)
        iter = np.nditer(temp, flags=['multi_index'])
        res = np.ones(len(iter))

        for i, it in enumerate(iter):
            mean = np.ones(gridsize)
            for j in range(gridsize):
                mean[j] = center[j] + (iter.multi_index[j] +
                                       0.5 - gridsize/2)*std*spread
            res[i] = -0.5*((x-mean).T @ np.linalg.inv(cov) @ (x-mean)) - \
                0.5 * np.log(2*(np.pi**2)*np.linalg.det(cov))

        return scipy.special.logsumexp(res)

    return log_posterior


def prior(params):
    return 1

# Defines whether to accept or reject the new sample


std = 5
dim = 32  # Highest dim number allowed is 32
cov = np.identity(dim, dtype=np.float32) * std

log_posterior = gen_fast_ndim_mog_posterior(5, cov, dim)


# def log_posterior(x, data):
#     return -0.5*(x.T @ np.linalg.inv(cov) @ x) - \
#         0.5 * np.log(np.linalg.det(2 * np.pi * cov))


x_init = 3 * np.random.normal(32)
exp_name = 'dim32_mog'

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


log_posterior(np.zeros(32), [])

if os.path.exists('log') is False:
    os.makedirs('log')

if os.path.exists('results') is False:
    os.makedirs('results')

RESULTS_DIR = 'results'
LOG_DIR = 'log'
