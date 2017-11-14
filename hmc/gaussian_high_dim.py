""" For Figures 5.6 and 5.7 from (Neal, 2010). """
from collections import defaultdict
import numpy as np
np.set_printoptions(suppress=True, linewidth=80)
import sys
import utils as U

# Matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
FIG_DIR = "draft_figures/"
title_size = 18
xsize = 15
ysize = 15
legend_size = 15
tick_size = 15


def f_H(q, p, cov_inv):
    """ The Hamiltonian function. See Section 5.3.3.1. """
    value = 0.5 * ((q.T).dot(cov_inv).dot(q) + (p.T).dot(p))
    return float(value)


def grad_U(q, cov_inv):
    """ The grad_U term. """
    return cov_inv.dot(q)


def leapfrog(current_q, cov_inv, eps, L, d):
    """ 
    Runs one leapfrog step. For this, think of q and p as the proposals, and the
    current_{q,p} stuff as the old stuff. Sorry if this is confusing.

    Parameters
    ----------
    current_q: The current position points, for which we may switch out of (if
    we accept the new proposed position) or otherwise we may stay here.

    Returns
    -------
    (q, p, a, h) where q is the new position variable, p was the final momentum
    variable (TODO: before or after negating?), a is a boolean indicating
    whether we accepted or not, and h is the final hamiltonian value.
    """
    q = np.copy(current_q)
    p = np.random.multivariate_normal(
            mean=np.zeros(d), cov=np.eye(d)).reshape((d,1)) 
    current_p = np.copy(p)

    p = p - (0.5*eps) * grad_U(q, cov_inv) 
    for i in range(L):
        # Full position update, then (except at the end) a half-momentum update.
        q = q + eps * p 
        if (i != L-1):
            p = p - eps * grad_U(q, cov_inv)
    p = p - (0.5*eps) * grad_U(q, cov_inv)
    p = -p

    # Determine whether we should accept or reject (and re-use).
    current_H = f_H(current_q, current_p, cov_inv)
    proposed_H = f_H(q, p, cov_inv)
    test_stat = (-proposed_H + current_H) # More numerically stable

    if (np.log(np.random.random()) < test_stat):
        return (q, 1, proposed_H)
    else:
        return (current_q, 0, current_H)
 

def run_samples_hmc(s, cov_inv, eps_range, N, L, d):
    """ Runs the example in Section 5.3.3.4.  """
    positions = []
    hamiltonians = []
    acc_rate = 0.0
    q = np.copy(s)
    
    for i in range(N):
        if (i % 100 == 0):
            print("HMC, sampling {}-th sample ...".format(i))

        # Randomly sample a new leapfrog step size.
        eps = np.random.uniform(*eps_range)
        q, a, ham = leapfrog(q, cov_inv, eps, L, d)
        acc_rate += a
        positions.append(np.squeeze(q))
        hamiltonians.append(ham)

    positions = np.squeeze(np.array(positions))
    hamiltonians = np.array(hamiltonians)
    acc_rate = acc_rate / N
    print("Finished HMC. Positions.shape: {}, acc rate: {:.3f}".format(
            positions.shape, acc_rate)) 
    return positions, acc_rate, hamiltonians


def run_samples_rw(s, cov_inv, std_range, N, d):
    """ 
    Runs a simple random walk, with provided covariance.  Here, the desired
    distribution is proprtional to exp(-0.5*q^T*inv(cov_distr)*q) so this is the
    value p(q) used in the ratio p(q')/p(q) for the acceptance test (let
    cov=cov_distr to simplify notation):

    p(q')/p(q) = exp(-0.5*(q')^T*inv(cov)*(q')) / exp(-0.5*q^T*inv(cov)*q)
               = exp( -0.5*(q')^T*inv(cov)*(q') + 0.5*q^T*inv(cov)*q )

    I'm pretty sure this is right! The results look good ... and we don't need
    any sort of momentum variables here since this isn't HMC.

    BTW, for this high dimensional example, Neal still uses the same standard
    deviation for the proposal for *all* of the coordinates.
    """
    positions = []
    acc_rate = 0.0
    curr_q = s

    for i in range(N):
        if (i % 5000 == 0):
            print("RW, sampling {}-th sample ...".format(i))

        # Randomly sample a new multivariate proposal distribution.
        std = np.random.uniform(*std_range)
        cov_rw = np.eye(d) * (std**2)
        prop_q = np.random.multivariate_normal(
                mean=curr_q.reshape(d,), cov=cov_rw).reshape((d,1))

        # Determine whether we should accept or reject.
        term_prop = float(0.5 * (prop_q.T).dot(cov_inv).dot(prop_q))
        term_curr = float(0.5 * (curr_q.T).dot(cov_inv).dot(curr_q))
        test_stat = (-term_prop + term_curr) # More numerically stable
        if (np.log(np.random.random()) < test_stat):
            curr_q = prop_q
            acc_rate += 1
        positions.append(curr_q)

    acc_rate = acc_rate / N
    positions = np.squeeze(np.array(positions))
    print("Finished RW. Positions.shape: {}, acc rate: {:.3f}".format(
            positions.shape, acc_rate)) 
    return positions, acc_rate


def plot_figure_5_6(hmc, rw, step):
    """ Plot Figure 5.6 to see coordinates. Might as well add the smallest. """
    fig, axarr = plt.subplots(2,2, figsize=(12,10))

    rw_first_coords  = rw['pos'][:,0][::step]
    rw_last_coords   = rw['pos'][:,-1][::step]
    hmc_first_coords = hmc['pos'][:,0]
    hmc_last_coords  = hmc['pos'][:,-1]
    assert len(rw_first_coords) == len(hmc_first_coords)
    xcoords = np.arange(len(rw_first_coords))

    # First coordinate
    axarr[0,0].plot(xcoords, rw_first_coords, 'ro')
    axarr[0,0].set_title("Random-Walk Metrop. ({:.3f})".format(rw['arate']), 
                         fontsize=title_size)
    axarr[0,1].plot(xcoords, hmc_first_coords, 'ro')
    axarr[0,1].set_title("Hamiltonian MC ({:.3f})".format(hmc['arate']), 
                         fontsize=title_size)

    # Last coordinate
    axarr[1,0].plot(xcoords, rw_last_coords, 'ro')
    axarr[1,0].set_title("Random-Walk Metrop. ({:.3f})".format(rw['arate']), 
                         fontsize=title_size)
    axarr[1,1].plot(xcoords, hmc_last_coords, 'ro')
    axarr[1,1].set_title("Hamiltonian MC ({:.3f})".format(hmc['arate']), 
                         fontsize=title_size)

    # Bells and whistles.
    for ii in range(2):
        for jj in range(2):
            if jj == 0:
                axarr[ii,jj].set_ylabel("Last Position Coordinate", fontsize=ysize)
                axarr[ii,jj].set_xlabel("Every {} Samples".format(step), fontsize=xsize)
            else:
                axarr[ii,jj].set_ylabel("First Position Coordinate", fontsize=ysize)
                axarr[ii,jj].set_xlabel("Every Sample", fontsize=xsize)
            axarr[ii,jj].set_ylim([-3.0, 3.0])
            axarr[ii,jj].tick_params(axis='x', labelsize=tick_size)
            axarr[ii,jj].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"high_dim_gaussian_5-6.png")


def plot_figure_5_7(hmc, rw, step, stdev_vec):
    """ Plot Figure 5.7 for an application of Bayesian models. :-) """
    fig, axarr = plt.subplots(2,2, figsize=(12,12))

    hmc_info = hmc['pos']
    n, d = hmc_info.shape
    indices = np.arange(n) * step
    rw_info = rw['pos'][indices]
    print("hmc positions shape: {}".format(hmc_info.shape))
    print("rw positions shape:  {}".format(rw_info.shape))
    rw_sample_mean  = np.mean(rw_info, axis=0)
    rw_sample_std   = np.std(rw_info, axis=0)
    hmc_sample_mean = np.mean(hmc_info, axis=0)
    hmc_sample_std  = np.std(hmc_info, axis=0)

    # First row = sample mean.
    axarr[0,0].plot(stdev_vec, rw_sample_mean, 'ro')
    axarr[0,0].set_title("Random-Walk Metropolis", fontsize=title_size)
    axarr[0,1].plot(stdev_vec, hmc_sample_mean, 'ro')
    axarr[0,1].set_title("Hamiltonian Monte Carlo", fontsize=title_size)

    # Second row = sample standard deviation.
    axarr[1,0].plot(stdev_vec, rw_sample_std, 'ro')
    axarr[1,0].set_title("Random-Walk Metropolis", fontsize=title_size)
    axarr[1,1].plot(stdev_vec, hmc_sample_std, 'ro')
    axarr[1,1].set_title("Hamiltonian Monte Carlo", fontsize=title_size)

    # Bells and whistles.
    for ii in range(2):
        for jj in range(2):
            if ii == 0:
                axarr[ii,jj].set_ylabel("Sample Mean of Coordinate", fontsize=ysize)
                axarr[ii,jj].set_ylim([-1,1])
            else:
                axarr[ii,jj].set_ylabel("Sample StDev of Coordinate", fontsize=ysize)
                axarr[ii,jj].set_ylim([0,1.2])
            axarr[ii,jj].set_xlabel("True StDev of Coordinate", fontsize=xsize)
            axarr[ii,jj].tick_params(axis='x', labelsize=tick_size)
            axarr[ii,jj].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"high_dim_gaussian_5-7.png")


if __name__ == "__main__":
    """ Define different covariances, but keep the same starting location. """
    dimension = 100
    stdev_vec = np.arange(1,dimension+1) / dimension
    variance_vec = stdev_vec ** 2 # GOTCHA. :-)
    cov_ham = np.diag(variance_vec)
    print("Here's the covariance matrix:\n{}".format(cov_ham))
    cov_inv = np.linalg.inv(cov_ham)
    # start_q = np.zeros((dimension,1))
    start_q = np.random.multivariate_normal(mean=np.zeros(dimension), 
                                            cov=np.eye(dimension))
    start_q = start_q.reshape((dimension,1)) 
    hmc, rw = {}, {}

    # Settings from (Neal, 2010).
    hmc_eps_range = (0.0104, 0.0156)
    rw_std_range = (0.0176, 0.0264)
    num_leapfrog = 150
    N_hmc = 1000 # It's 1000 but do smaller to test.
    N_rw = N_hmc * num_leapfrog

    hmc['pos'], hmc['arate'], hmc['ham'] = run_samples_hmc(
            s=np.copy(start_q),
            cov_inv=cov_inv,
            eps_range=hmc_eps_range,
            N=N_hmc,
            L=num_leapfrog,
            d=dimension)
    rw['pos'], rw['arate'] = run_samples_rw(
            s=np.copy(start_q),
            cov_inv=cov_inv,
            std_range=rw_std_range,
            N=N_rw,
            d=dimension)
    plot_figure_5_6(hmc, rw, step=num_leapfrog)
    plot_figure_5_7(hmc, rw, step=num_leapfrog, stdev_vec=stdev_vec)
