"""
Simple Hamiltonian. Based on Section 5.3.3.2 in Radford Neal's article, 2010.
This time, we actually sample multiple times (each single sample needs its own
sequence of leapfrog steps).
"""
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


def f_H(q, p, cov):
    """ The Hamiltonian function. See Section 5.3.3.1. """
    value = 0.5 * ((q.T).dot(np.linalg.inv(cov)).dot(q) + (p.T).dot(p))
    return float(value)


def grad_U(q, cov):
    """ The grad_U term. """
    return (np.linalg.inv(cov)).dot(q)


def leapfrog(current_q, cov, eps, L):
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
            mean=np.array([0,0]), cov=np.eye(2)).reshape((2,1)) 
    current_p = np.copy(p)

    p = p - (0.5*eps) * grad_U(q, cov) 
    for i in range(L):
        # Full position update, then (except at the end) a half-momentum update.
        q = q + eps * p 
        if (i != L-1):
            p = p - eps * grad_U(q, cov)
    p = p - (0.5*eps) * grad_U(q, cov)
    p = -p

    # Determine whether we should accept or reject (and re-use).
    current_H = f_H(current_q, current_p, cov)
    proposed_H = f_H(q, p, cov)
    test_stat = np.exp(-proposed_H + current_H)

    if (np.random.random() < test_stat):
        return (q, None, 1, proposed_H)
    else:
        return (current_q, None, 0, current_H)


def run_samples_hmc(s, covariance, N, eps, L):
    """
    Runs the example in Section 5.3.3.2 in Neal's paper using Hamiltonian Monte
    Carlo. [Unfortunately he doesn't explain which starting q he used.]
    """
    positions = []
    momentums = []
    hamiltonians = []
    acc_rate = 0.0
    q = np.copy(s)
    
    for i in range(N):
        q, mom, a, ham = leapfrog(q, covariance, eps, L)
        acc_rate += a
        positions.append(np.squeeze(q))
        momentums.append(np.squeeze(mom))
        hamiltonians.append(ham)

    positions = np.array(positions)
    momentums = np.array(momentums)
    hamiltonians = np.array(hamiltonians)
    acc_rate = acc_rate / N
    return positions, momentums, acc_rate, hamiltonians


def run_samples_rw(s, cov_rw, cov_distr, N):
    """ 
    Runs a simple random walk, with provided covariance.  Here, the desired
    distribution is proprtional to exp(-0.5*q^T*inv(cov_distr)*q) so this is the
    value p(q) used in the ratio p(q')/p(q) for the acceptance test (let
    cov=cov_distr to simplify notation):

    p(q')/p(q) = exp(-0.5*(q')^T*inv(cov)*(q')) / exp(-0.5*q^T*inv(cov)*q)
               = exp( -0.5*(q')^T*inv(cov)*(q') + 0.5*q^T*inv(cov)*q )

    I'm pretty sure this is right! The results look good ... and we don't need
    any sort of momentum variables here since this isn't HMC.
    """
    positions = []
    acc_rate = 0.0
    curr_q = s
    cov_inv = np.linalg.inv(cov_distr)

    for i in range(N):
        prop_q = np.random.multivariate_normal(
                mean=curr_q.reshape(2,), cov=cov_rw).reshape((2,1))
        term_prop = float(0.5 * (prop_q.T).dot(cov_inv).dot(prop_q))
        term_curr = float(0.5 * (curr_q.T).dot(cov_inv).dot(curr_q))
        test_stat = (-term_prop + term_curr) # More numerically stable
        if (np.log(np.random.random()) < test_stat):
            curr_q = prop_q
            acc_rate += 1
        positions.append(curr_q)

    acc_rate = acc_rate / N
    return np.array(positions), acc_rate


def plot(hmc, rw, cov_ham):
    """  For plotting. """
    fig, axarr = plt.subplots(1,3, figsize=(15,4.5))

    # Get ellipses of the target distribution.
    for ii in range(2):
        U.plot_cov_ellipse(cov=cov_ham, pos=[0,0], nstd=1, ax=axarr[ii],
                alpha=0.25, color='blue', ls='dashed', lw=3)

    axarr[0].plot(rw['pos'][:,0][::20], rw['pos'][:,1][::20], '-ro')
    axarr[0].set_title("Random-walk (Acc: {:.3f})".format(rw['arate']), 
                       fontsize=title_size)
    axarr[1].plot(hmc['pos'][:,0], hmc['pos'][:,1], '-ro')
    axarr[1].set_title("Hamiltonian MC (Acc: {:.3f})".format(hmc['arate']), 
                       fontsize=title_size)
    axarr[2].plot(hmc['ham'], '-ro')
    axarr[2].set_title("Hamiltonians", fontsize=title_size)

    # Bells and whistles.
    for ii in range(3):
        if ii < 2:
            axarr[ii].set_xlim([-2.5,2.5])
            axarr[ii].set_ylim([-2.5,2.5])
        axarr[ii].tick_params(axis='x', labelsize=tick_size)
        axarr[ii].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"bivariate_gaussians_many_samples.png")


def plot_figure_5_5(hmc, rw):
    """ Plot Figure 5.5 to see coordinates, and add the second ones. """
    fig, axarr = plt.subplots(2,2, figsize=(10,8))

    rw_first_coords = rw['pos'][:,0][::20]
    hmc_first_coords = hmc['pos'][:,0]
    rw_second_coords = rw['pos'][:,1][::20]
    hmc_second_coords = hmc['pos'][:,1]

    assert len(rw_first_coords) == len(hmc_first_coords)
    xcoords = np.arange(len(rw_first_coords))

    axarr[0,0].plot(xcoords, rw_first_coords, '-ro')
    axarr[0,0].set_title("Random-Walk Metropolis", fontsize=title_size)
    axarr[0,0].set_xlabel("Every 20 Samples", fontsize=xsize)
    axarr[0,1].plot(xcoords, hmc_first_coords, '-ro')
    axarr[0,1].set_title("Hamiltonian Monte Carlo", fontsize=title_size)
    axarr[0,1].set_xlabel("Every Sample", fontsize=xsize)

    axarr[1,0].plot(xcoords, rw_second_coords, '-ro')
    axarr[1,0].set_title("Random-Walk Metropolis", fontsize=title_size)
    axarr[1,0].set_xlabel("Every 20 Samples", fontsize=xsize)
    axarr[1,1].plot(xcoords, hmc_second_coords, '-ro')
    axarr[1,1].set_title("Hamiltonian Monte Carlo", fontsize=title_size)
    axarr[1,1].set_xlabel("Every Sample", fontsize=xsize)

    # Bells and whistles.
    for ii in range(2):
        for jj in range(2):
            if ii == 0:
                axarr[ii,jj].set_ylabel("First Position Coordinate", fontsize=ysize)
            else:
                axarr[ii,jj].set_ylabel("Second Position Coordinate", fontsize=ysize)
            axarr[ii,jj].set_ylim([-3.0, 3.0])
            axarr[ii,jj].tick_params(axis='x', labelsize=tick_size)
            axarr[ii,jj].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"bivariate_gaussians_fig5-5.png")


if __name__ == "__main__":
    """ Define different covariances, but keep the same starting location. """
    cov_ham = np.array([[1., 0.98],[0.98, 1.]])
    cov_rw  = np.array([[1., 0.],[0., 1.]]) * (0.18**2)
    start_q = np.array([[0.],[0.]])
    hmc, rw = {}, {}

    hmc['pos'], hmc['mom'], hmc['arate'], hmc['ham'] = run_samples_hmc(
            s=np.copy(start_q),
            covariance=cov_ham,
            N=20, 
            eps=0.18, 
            L=20)
    rw['pos'], rw['arate'] = run_samples_rw(
            s=np.copy(start_q),
            cov_rw=cov_rw,
            cov_distr=cov_ham,
            N=400)
    plot(hmc, rw, cov_ham)

    # Run these more times to get statistics.
    hmc_arates_l = []
    rw_arates_l = []
    for _ in range(20):
        _, _, hmc_arate, _ = run_samples_hmc(
                s=np.copy(start_q), covariance=cov_ham, N=20, eps=0.18, L=20)
        _, rw_arate = run_samples_rw(
            s=np.copy(start_q), cov_rw=cov_rw, cov_distr=cov_ham, N=400)
        hmc_arates_l.append(hmc_arate)
        rw_arates_l.append(rw_arate)
    print("HMC accept rate: {:.3f} +/- {:.3f}".format(
            np.mean(hmc_arates_l), np.std(hmc_arates_l)))
    print("RW accept rate: {:.3f} +/- {:.3f}".format(
            np.mean(rw_arates_l), np.std(rw_arates_l)))

    # Now do Figure 5.5 which runs for longer and plots magnitudes.
    hmc, rw = {}, {}
    hmc['pos'], hmc['mom'], hmc['arate'], hmc['ham'] = run_samples_hmc(
            s=np.copy(start_q), covariance=cov_ham, N=200, eps=0.18, L=20)
    rw['pos'], rw['arate'] = run_samples_rw(
            s=np.copy(start_q), cov_rw=cov_rw, cov_distr=cov_ham, N=4000)
    plot_figure_5_5(hmc, rw)
