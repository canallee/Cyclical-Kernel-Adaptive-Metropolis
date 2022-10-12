import numpy as np
import random
from numpy.random import multivariate_normal as randN
from scipy.stats import multivariate_normal
from scipy.special import gamma, kv
import scipy
import scipy.stats
from global_data import *
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(1234)
###############################################################################################################
# Computes gradient of a guassian kernel


def get_grad_Guassian(bandwidth=1.0):
    def grad_func(x, x_prime):
        # guassian kernel function
        def k(x, x_prime, bandwidth):
            tmp_1 = -1 * np.linalg.norm(x-x_prime)**2
            tmp_2 = 2*bandwidth**2
            return np.exp(tmp_1/tmp_2)
        k_x_xp = k(x, x_prime, bandwidth)
        return 1/(bandwidth**2)*k_x_xp*(x_prime-x)
    return grad_func

# Computes gradient of a linear kernel


def get_grad_Linear():
    def grad_func(x, x_prime):
        return x_prime
    return grad_func

# Computes gradient of a Matérn kernel


def get_grad_Matern(l=1.0, v=1.5):
    def grad_func(x, x_prime):
        # Matérn kernel function
        def k(x, x_prime, l, v):
            scale = 1 / (gamma(v) * 2**(v-1))
            tmp = (np.sqrt(2*v)/l) * np.linalg.norm(x-x_prime)
            if tmp == 0:
                return 1.0
            else:
                return scale * tmp**v * kv(v, tmp)
        tmp_out = v/(l**2 * (v-1))
        return tmp_out * k(x, x_prime, l, v-1) * (x_prime - x)
    return grad_func

###############################################################################################################
# computes a kernel gradient matrix d x m, m = min{subsample_size, len(history)}
# ////
# x:              current position, of dim=d
# history:        chain history of accepted positions d x t
# gradK:          function to compute the gradient of a kernel function w.r.t x
# idx:            idx for the subsample of chain history
# ////
# Returns:        M, the kernel gradient matrix d x m


def get_kernel_grad_matrix(x, history, idx, gradK):
    d, t = history.shape
    m = len(idx)
    M = np.zeros([d, m])
    count = 0
    for i in range(m):
        M[:, i] = 2 * gradK(x, history[:, idx[i]])
        count += 1
    return M

# get centraling matrix H
# m:              min{subsample_size, len(history)}
# Returns:        centraling matrix H


def get_centraling_matrix(m):
    I = np.identity(m)
    ones = np.ones([m, m])
    H = I - ones/m
    return H

# get idx for subsample z, resample z with decreasing probability {p_t}
# iterations:       total number of iterations, p_0 = 1.0, p_iterations = 0.01
# decay_rate:       controls p_t to be an exponential decay rate, with decay_rate in (0, 1) controls half-life
# subsample_size:   number of subsamples from chain history
# w_replacement:    subsample with or without replacement
# returns:          a function that takes in 1) current idx list, 2) history
#                   and generates a new idx list with prob p_t


def idx_scheduler(iterations, decay_rate=0.5, subsample_size=50, w_replacement=False):
    t_half = iterations/(2/decay_rate)
    gamma = np.log(2)/t_half

    def idx_func(idx, history):
        if len(idx) == 0:
            return idx
        _, current_t = history.shape
        p_t = np.exp(-gamma * current_t)
        m = min(subsample_size, current_t)
        dice = np.random.uniform(0, 1)
        if dice > p_t:
            idx = np.random.choice(current_t, m, replace=w_replacement)
        return idx
    return idx_func

# noise scheduler, decaying from initial_noise -> final_noise
# iterations:       total number of iterations, p_0 = 1.0, p_iterations = 0.01
# initial_noise:    starting value for noise in KAM proposal
# final_noise:      final value for noise
# decay_rate:       controls the rate of decay, value in (0,1)
# returns:          a function that takes in current iteration t, and returns noise for that t


def decaying_noise_scheduler(iterations, initial_noise=4, final_noise=0.2, decay_rate=0.55):
    tmp = (final_noise/initial_noise) ** (1/decay_rate)
    b = (tmp*iterations) / (1 - tmp)
    a = initial_noise * b ** decay_rate

    def noise_func(current_t):
        return a * (b + current_t)**(-decay_rate)
    return noise_func

# noise scheduler, constant noise


def constant_noise_scheduler(noise=0.2):
    def noise_func(current_t):
        return noise
    return noise_func

# Propose the next position x' with KAM proposal, also computes the proposal ratio
# ////
# x:               current position, of dim=d
# history:         chain history of accepted positions d x t
# gradK:           function to compute the gradient of a kernel function w.r.t x
# stepsize:        current stepsize
# noise:           noise term gamma
# idx:             idx for subsample z of chain history
# ////
# Returns:         the proposed position x', proposal_ratio q(x|x')/q(x'|x), [optionally] cov_forward


def KAM_proposal(x, history, gradK, stepsize, idx, noise, return_cov=False):
    d, t = history.shape
    if t == 0:
        x_prime = randN(x, noise**2 * np.identity(d))
        return x_prime, 1
    else:
        m = len(idx)
        M = get_kernel_grad_matrix(x, history, idx, gradK)
        H = get_centraling_matrix(m)
        cov_forward = noise**2 * np.identity(d) + stepsize**2 * (M @ H @ M.T)
        var_forward = multivariate_normal(
            mean=x, cov=cov_forward, allow_singular=True)
        # propose next position x'
        x_prime = randN(x, cov_forward)
        # compute q(x'|x)
        q_forward = var_forward.pdf(x_prime)
        M_prime = get_kernel_grad_matrix(x_prime, history, idx, gradK)
        cov_reverse = noise**2 * np.identity(d) + stepsize**2 * (M_prime @ H @ M_prime.T)
        var_reverse = multivariate_normal(
            mean=x_prime, cov=cov_reverse, allow_singular=True)
        # compute q(x|x')
        q_reverse = var_reverse.pdf(x)
        if return_cov:
            return x_prime, q_reverse/q_forward, cov_forward
        else:
            return x_prime, q_reverse/q_forward


def KAM(likelihood_computer, prior, x_init, iterations, burn_in, data, acceptance_rule, stepsize_init,
        gradK, idx_scheduler, noise_scheduler, eps=0.75, optim_acc=0.234):
    # initialization
    x = np.array(x_init)
    stepsize = stepsize_init
    history = np.empty([len(x), 0])
    samples = []
    accept_acc = 0
    idx = np.empty(0)
    samples_index_array = []
    stepsize_array = []
    period_after_burn_in = 0
    # begin sampling
    for i in range(iterations):
        period_start_after_burnin = time.time()

        idx = idx_scheduler(idx, history)
        noise = noise_scheduler(i)
        x_prime, proposal_ratio = KAM_proposal(
            x, history, gradK, stepsize, idx, noise)
        log_pi = likelihood_computer(x, data) + np.log(prior(x))
        log_pi_prime = likelihood_computer(
            x_prime, data) + np.log(prior(x_prime))
        # MH step
        sig, acc_ratio = acceptance_rule(log_pi, log_pi_prime, proposal_ratio)
        if sig:
            x = x_prime
        if i >= burn_in:
            samples.append(x)
            if sig == 1:
                accept_acc += 1
        history = np.hstack((history, x.reshape((-1, 1))))
        # Adapting stepsize
        eta = 1 / (1+i)**eps
        stepsize = np.exp(np.log(stepsize) + eta*(acc_ratio - optim_acc))
        stepsize_array.append(stepsize)

        if i > 0 and i % 100000 == 0:
            with open(f"{LOG_DIR}{os.sep}KAM-{i}", 'w') as f:
                f.write('')

        if i > burn_in:
            period_after_burn_in += (time.time() - period_start_after_burnin)

        if i > burn_in and period_after_burn_in >= 10:
            period_after_burn_in = 0
            samples_index_array.append(len(samples))

    avg_acc = accept_acc / (iterations - burn_in)
    return np.array(samples), avg_acc, history, np.array(samples_index_array), np.array(stepsize_array)


iterations = 400000
burnin = 160000
initial_stepsize = 4*2.38**2/2
# get externel functions
gradk = get_grad_Matern(l=2, v=4)
idx_schedule = idx_scheduler(iterations, subsample_size=30, decay_rate=0.4)
noise_schedule = decaying_noise_scheduler(
    iterations, initial_noise=4, final_noise=1, decay_rate=0.55)
# begin training
samples_KAM, avg_acc, history, samples_index_array_KAM, stepsize_array_KAM = KAM(log_posterior, prior, x_init, iterations, burnin, [],
                                                                                 acceptance, initial_stepsize, gradk, idx_schedule, noise_schedule,
                                                                                 eps=2, optim_acc=0.234)

np.save(f'{RESULTS_DIR}{os.sep}samples_KAM_{exp_name}.npy', samples_KAM)
np.save(f'{RESULTS_DIR}{os.sep}samples_index_array_KAM_{exp_name}.npy', samples_index_array_KAM)
np.save(f'{RESULTS_DIR}{os.sep}stepsize_array_KAM_{exp_name}.npy', stepsize_array_KAM)
