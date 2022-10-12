import numpy as np
import os


def counter_posterior(x, data):
    mean1 = np.array([-8, 0])
    cov1 = np.array([[0.5, 0], [0, 0.5]], dtype=np.float32)
    mean2 = np.array([8, 0])
    cov2 = np.array([[2, 0], [0, 2]], dtype=np.float32)
    res = 0
    res += np.exp(-0.5*(x-mean1).T@np.linalg.inv(cov1)@(x-mean1))/np.sqrt(2*(np.pi**2)*np.linalg.det(cov1))/2
    res += np.exp(-0.5*(x-mean2).T@np.linalg.inv(cov2)@(x-mean2))/np.sqrt(2*(np.pi**2)*np.linalg.det(cov2))
    return np.log(res)


def prior(params):
    return 1

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
X_INIT = np.array([-8, 0])