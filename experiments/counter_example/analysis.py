# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from multi_ESS import multiESS
import statsmodels.api as sm
import scipy

# %%
samples_MH = np.load(
    f'results{os.sep}samples_MH_counter_example.npy')
samples_AM = np.load(
    f'results{os.sep}samples_AM_counter_example.npy')
samples_GAM = np.load(
    f'results{os.sep}samples_GAM_counter_example.npy')
samples_KAM = np.load(
    f'results{os.sep}samples_KAM_counter_example.npy')
samples_RBAM = np.load(
    f'results{os.sep}samples_RBAM_counter_example.npy')
samples_cKAM = np.load(
    f'results{os.sep}samples_cKAM_counter_example.npy')

samples_MH_index = np.load(
    f'results{os.sep}samples_index_array_MH.npy')
samples_AM_index = np.load(
    f'results{os.sep}samples_index_array_AM.npy')
samples_GAM_index = np.load(
    f'results{os.sep}samples_index_array_GAM.npy')
samples_KAM_index = np.load(
    f'results{os.sep}samples_index_array_KAM.npy')
samples_RBAM_index = np.load(
    f'results{os.sep}samples_index_array_RBAM.npy')
samples_cKAM_index = np.load(
    f'results{os.sep}samples_index_array_cKAM.npy')

stepsize_array_KAM = np.load(
    f'results{os.sep}stepsize_array_KAM.npy')
stepsize_array_cKAM = np.load(
    f'results{os.sep}stepsize_array_cKAM.npy')

# %%


def compute_kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = 0.0
    for p, q in zip(p_probs, q_probs):
        if p == 0 or q == 0:
            continue
        kl_div += p * np.log(p / q)
    return kl_div


def symKL(h1, h2):
    return compute_kl_divergence(h1, h2) + compute_kl_divergence(h2, h1)


def compute_histogram2d(data, bins, value_range):
    ret = np.histogram2d(data[:, 0], data[:, 1], bins=(
        bins, bins), range=value_range)[0].flatten()
    return ret / np.sum(ret)


def compute_true_histogram2d(log_posterior, x_list, data):
    n, d = x_list.shape
    log_P = np.zeros(n)
    for i, x in enumerate(x_list):
        log_P[i] = log_posterior(x, data)
    normalized_term = scipy.special.logsumexp(log_P.flatten())
    ret = np.exp(log_P - normalized_term).flatten()
    # ret[ret < 1e-7] = 0
    return ret


def plot_autocorr(samples, title, max_lag=100, gap_to_show=5, y_max=0.5, y_min=-0.1):
    n, d = samples.shape
    A_lag_d = np.zeros((d, max_lag))
    for dim in range(d):
        A_lag_d[dim, :] = sm.tsa.acf(samples[:, dim], nlags=max_lag)[1:]
    lags_to_show = np.arange(5, max_lag, gap_to_show)
    plt.boxplot(A_lag_d[:, lags_to_show], labels=list(lags_to_show))
    plt.title(f'Auto-correlation for All Dimensions in the Samples - {title}')
    plt.xlabel('lag')
    plt.ylabel('Auto-correlation')
    plt.show()


def compute_multi_ESS(samples, sample_index_array):
    ESS = []
    for i in sample_index_array:
        ESS.append(multiESS(samples[:i]))
    return ESS


def compute_KL(correct_histogram2d, samples, sample_index_array, bins, value_range):
    KL = []
    for i in sample_index_array:
        samples_histogram2d = compute_histogram2d(
            samples[:i], bins, value_range)
        KL.append(symKL(correct_histogram2d, samples_histogram2d))
    return KL


# %%
def counter_posterior(x, data):
    mean1 = np.array([-8, 0])
    cov1 = np.array([[0.5, 0], [0, 0.5]], dtype=np.float32)
    mean2 = np.array([8, 0])
    cov2 = np.array([[2, 0], [0, 2]], dtype=np.float32)
    res = 0
    res += np.exp(-0.5*(x-mean1).T@np.linalg.inv(cov1)@(x-mean1)) / \
        np.sqrt(2*(np.pi**2)*np.linalg.det(cov1))/2
    res += np.exp(-0.5*(x-mean2).T@np.linalg.inv(cov2)@(x-mean2)) / \
        np.sqrt(2*(np.pi**2)*np.linalg.det(cov2))
    return np.log(res)


x_low, x_high = -17, 17
y_low, y_high = -12, 12
value_range = [[x_low, x_high], [y_low, y_high]]
bin_cnt = 600
display_low = 0
display_high = 100
log_interval = 10

# %%


def true_1d_marginalized(log_posterior, x_list, y_list, data):
    log_P = np.zeros((len(x_list), len(y_list)))
    for i, x in enumerate(x_list):
        for j, y in enumerate(y_list):
            log_P[i, j] = log_posterior(np.array([x, y]), data)
    normalized_term = scipy.special.logsumexp(log_P.flatten())
    ret = np.exp(log_P - normalized_term)
    return np.sum(ret, axis=1)


# %%
def samples_1d_marginalized(samples, x_low, x_high, bins):
    ret = np.histogram(samples[:, 0], bins, range=[x_low, x_high])[0]
    return ret / np.sum(ret)


# %%
x_edges = np.linspace(x_low, x_high, bin_cnt)
y_edges = np.linspace(y_low, y_high, bin_cnt)
xxyy = np.array(np.meshgrid(x_edges, y_edges)).T.reshape(-1, 2)

true_histogram = compute_true_histogram2d(counter_posterior, xxyy, [])
true_1d = true_1d_marginalized(counter_posterior,
                               np.linspace(x_low, x_high, bin_cnt),
                               np.linspace(y_low, y_high, bin_cnt),
                               []
                               )


# %%
MH_KL = compute_KL(true_histogram, samples_MH,
                   samples_MH_index, bin_cnt, value_range)
AM_KL = compute_KL(true_histogram, samples_AM,
                   samples_AM_index, bin_cnt, value_range)
GAM_KL = compute_KL(true_histogram, samples_GAM,
                    samples_GAM_index, bin_cnt, value_range)
RBAM_KL = compute_KL(true_histogram, samples_RBAM,
                     samples_RBAM_index, bin_cnt, value_range)
KAM_KL = compute_KL(true_histogram, samples_KAM,
                    samples_KAM_index, bin_cnt, value_range)
cKAM_KL = compute_KL(true_histogram, samples_cKAM,
                     samples_cKAM_index, bin_cnt, value_range)
# samples_MH_index.shape


# %%
plt.figure(figsize=(10, 5), dpi=200)
plt.plot(log_interval * np.arange(display_low, display_high),
         MH_KL[display_low:display_high], label='MH')
plt.plot(log_interval * np.arange(display_low, display_high),
         AM_KL[display_low:display_high], label='AM')
plt.plot(log_interval * np.arange(display_low, display_high),
         GAM_KL[display_low:display_high], label='GAM')
plt.plot(log_interval * np.arange(display_low, display_high),
         KAM_KL[display_low:display_high], label='KAM')
plt.plot(log_interval * np.arange(display_low, display_high),
         RBAM_KL[display_low:display_high], label='RBAM')
plt.plot(log_interval * np.arange(display_low, display_high),
         cKAM_KL[display_low:display_high], label='cKAM')
plt.title('counter example')
plt.xlabel('seconds after burnin')
plt.ylabel('sym KL divergence')
plt.legend()
plt.savefig('counter_example_KL.png')


# %%
MH_1d = samples_1d_marginalized(samples_MH, x_low, x_high, bin_cnt)
AM_1d = samples_1d_marginalized(samples_AM, x_low, x_high, bin_cnt)
GAM_1d = samples_1d_marginalized(samples_GAM, x_low, x_high, bin_cnt)
RBAM_1d = samples_1d_marginalized(samples_RBAM, x_low, x_high, bin_cnt)
KAM_1d = samples_1d_marginalized(samples_KAM, x_low, x_high, bin_cnt)
cKAM_1d = samples_1d_marginalized(samples_cKAM, x_low, x_high, bin_cnt)


# %%
plt.figure(figsize=(10, 5), dpi=1000)
plt.plot(np.linspace(x_low, x_high, bin_cnt),
         true_1d, label='true counter posterior')
plt.plot(np.linspace(x_low, x_high, bin_cnt), MH_1d, '--', label='MH')
plt.plot(np.linspace(x_low, x_high, bin_cnt), AM_1d, '--', label='AM')
plt.plot(np.linspace(x_low, x_high, bin_cnt), GAM_1d, '--', label='GAM')
plt.plot(np.linspace(x_low, x_high, bin_cnt), KAM_1d, '--', label='KAM')
plt.plot(np.linspace(x_low, x_high, bin_cnt), RBAM_1d, '--', label='RBAM')
plt.plot(np.linspace(x_low, x_high, bin_cnt), cKAM_1d, '--', label='cKAM')
plt.title('counter example: marginal prob on dim 1')
plt.xlabel('x axis')
plt.ylabel('marginalized 1d prob')
plt.legend()
plt.savefig('counter_example_marginalized_1d.png')


# %%
MH_ESS = compute_multi_ESS(samples_MH, samples_MH_index)
AM_ESS = compute_multi_ESS(samples_AM, samples_AM_index)
GAM_ESS = compute_multi_ESS(samples_GAM, samples_GAM_index)
RBAM_ESS = compute_multi_ESS(samples_RBAM, samples_RBAM_index)
KAM_ESS = compute_multi_ESS(samples_KAM, samples_KAM_index)
cKAM_ESS = compute_multi_ESS(samples_cKAM, samples_cKAM_index)


# %%
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(log_interval * np.arange(display_low, display_high),
         MH_ESS[display_low:display_high], label='MH')
plt.plot(log_interval * np.arange(display_low, display_high),
         AM_ESS[display_low:display_high], label='AM')
plt.plot(log_interval * np.arange(display_low, display_high),
         GAM_ESS[display_low:display_high], label='GAM')
plt.plot(log_interval * np.arange(display_low, display_high),
         KAM_ESS[display_low:display_high], label='KAM')
plt.plot(log_interval * np.arange(display_low, display_high),
         RBAM_ESS[display_low:display_high], label='RBAM')
plt.plot(log_interval * np.arange(display_low, display_high),
         cKAM_ESS[display_low:display_high], label='cKAM')
plt.title('counter example')
plt.xlabel('seconds after burnin')
plt.ylabel('multi ESS')
plt.legend()
plt.savefig('counter_example_ESS.png')


# %%
def make_density_plot(samples, title, bins, value_range):
    plt.figure()
    plt.hist2d(samples[:, 0], samples[:, 1], (bins, bins), range=value_range)
    plt.title(title)
    plt.savefig(f"counter_example-{title}-density.png")


make_density_plot(samples_MH, "MH", bin_cnt, value_range)
make_density_plot(samples_AM, "AM", bin_cnt, value_range)
make_density_plot(samples_GAM, "GAM", bin_cnt, value_range)
make_density_plot(samples_KAM, "KAM", bin_cnt, value_range)
make_density_plot(samples_cKAM, "cKAM", bin_cnt, value_range)
make_density_plot(samples_RBAM, "RBAM", bin_cnt, value_range)

# %%
