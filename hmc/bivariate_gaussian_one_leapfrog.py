"""
Attempts to reproduce Figure 5.3 in (Neal, 2010) along with accompanying
extensions. To be clear, this is NOT HMC, but just one ITERATION in hmc.

Notation: q = position, p = momentum. Yeah, sorry I don't know why it's like
that, confusing.  
"""
from collections import defaultdict
import numpy as np
np.set_printoptions(suppress=True, linewidth=80)
import sys

# Matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
FIG_DIR = "draft_figures/"
title_size = 18
xsize = 15
ysize = 15
legend_size = 15
tick_size = 15


def proc(pylist):
    return np.squeeze(np.array(pylist))


def f_H(q, p, cov):
    """ The Hamiltonian function. See Section 5.3.3.1. """
    value = 0.5 * ((q.T).dot(np.linalg.inv(cov)).dot(q) + (p.T).dot(p))
    return float(value)


def grad_U(q, cov):
    """ The grad_U term. """
    return (np.linalg.inv(cov)).dot(q)


def do_one_sample(cfg):
    """
    Runs the example in Section 5.3.3.1 in Neal 2010, using the provided initial
    states. States should be column vectors. Assumes we have a configuration
    which gives us what we need. In reality, though, we'd be sampling for p
    instead of choosing p ourselves. And we shouldn't hard-code these choices.
    """
    eps = cfg['eps']
    cov = np.array([[cfg['stdev'], cfg['covariance']],
                    [cfg['covariance'], cfg['stdev']]]
    )
    q = cfg['qstart']
    p = cfg['pstart']
    positions = [ np.copy(q) ] # not p ;-)
    momentums = [ np.copy(p) ] 
    hamiltonians = [ f_H(q, p, cov) ]

    # Now do the leapfrog stuff.
    p = p - (0.5*eps) * grad_U(q, cov) 

    for i in range(cfg['L']):
        # Do a full position update, here grad_K(p) = p.
        q = q + eps * p
        positions.append(q)

        # Then do a full momentum step (i.e., combining two half-steps).
        # NORMALLY we wouldn't split the two half-steps like I did, but I split
        # them to get my plot to match Neal's, because technically we want one
        # half momentum step, one full position step, then a second half
        # momentum step, and that's what we'd be plotting since that is
        # considered one "new" set of (position,momentum) variables. For
        # practical usage, we'd just combine the two steps unless I really
        # needed to see the exact positions for perhaps debugging purposes.

        if (i != cfg['L']-1):
            p = p - (0.5*eps) * grad_U(q, cov)
            momentums.append(p)
            hamiltonians.append(f_H(q, p, cov))
            p = p - (0.5*eps) * grad_U(q, cov)

    # Half-step for the momentum at the end.
    p = p - (0.5*eps) * grad_U(q, cov)
    momentums.append(p)
    hamiltonians.append(f_H(q, p, cov))

    # Not needed here but w/e. We'd also do an MH test here if this were part of
    # a full-blown HMC method.
    p = -p 

    return proc(positions), proc(momentums), proc(hamiltonians)


def plot(positions, momentums, hamiltonians, cfg):
    """  Creates plots to match Neal's figure. """
    L = cfg['L']
    eps = cfg['eps']
    fig, axarr = plt.subplots(1,3, figsize=(15,4.5))

    axarr[0].plot(positions[:,0], positions[:,1], '-ro')
    axarr[0].set_title("Positions, eps={}, L={}".format(eps, L), 
                       fontsize=title_size)
    axarr[0].set_xlim([-2.5,2.5])
    axarr[0].set_ylim([-2.5,2.5])

    axarr[1].plot(momentums[:,0], momentums[:,1], '-ro')
    axarr[1].set_title("Momentums, eps={}, L={}".format(eps, L), 
                       fontsize=title_size)
    axarr[1].set_xlim([-2.5,2.5])
    axarr[1].set_ylim([-2.5,2.5])

    axarr[2].plot(hamiltonians, '-ro')
    axarr[2].set_title("Hamiltonian, eps={}, L={}".format(eps, L), 
                       fontsize=title_size)
    axarr[2].axhline(y=hamiltonians[0], color='b', linestyle='-')

    for ii in range(3):
        axarr[ii].tick_params(axis='x', labelsize=tick_size)
        axarr[ii].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"bivariate_gaussians_one_leapfrog_traj.png")


def plot_groups(stats, eps_values):
    """  Creates plots to match Neal's figure. """
    nrows = len(eps_values)
    ncols = 3
    fig, axarr = plt.subplots(nrows, ncols, figsize=(15,5*nrows))

    for i in range(nrows):
        positions = stats['all_pos'][i]
        momentums = stats['all_mom'][i]
        hamiltonians = stats['all_ham'][i]
        eps = eps_values[i]

        axarr[i,0].plot(positions[:,0], positions[:,1], '-ro')
        axarr[i,1].plot(momentums[:,0], momentums[:,1], '-ro')
        axarr[i,2].plot(hamiltonians, '-ro')

        axarr[i,0].set_title("Positions, eps={}".format(eps), 
                             fontsize=title_size)
        axarr[i,1].set_title("Momentums, eps={}".format(eps), 
                             fontsize=title_size)
        axarr[i,2].set_title("Hamiltonian, eps={}".format(eps), 
                             fontsize=title_size)
        axarr[i,2].axhline(y=hamiltonians[0], color='b', linestyle='-')

        for j in range(ncols):
            if j < 2:
                axarr[i,j].set_xlim([-2.5,2.5])
                axarr[i,j].set_ylim([-2.5,2.5])
            else:
                pass
            axarr[i,j].tick_params(axis='x', labelsize=tick_size)
            axarr[i,j].tick_params(axis='y', labelsize=tick_size)

    plt.tight_layout()
    plt.savefig(FIG_DIR+"bivariate_gaussians_one_leapfrog_investigation.png")


if __name__ == "__main__":
    # Reproduce Figure 5.3 from (Neal, 2010).
    cfg = {'eps': 0.25, 
           'L': 25,
           'covariance': 0.95,
           'stdev': 1.0,
           'qstart': np.array([[-1.50],[-1.55]]),
           'pstart': np.array([[-1.],[1.]])
    }
    positions, momentums, hamiltonians = do_one_sample(cfg)
    print("hamiltonians:\n{}".format(hamiltonians))
    plot(positions, momentums, hamiltonians, cfg)

    # Some additional investigation.
    eps_values = [0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.44, 0.45]
    stats = defaultdict(list)
    for e in eps_values:
        cfg['eps'] = e
        positions, momentums, hamiltonians = do_one_sample(cfg)
        stats['all_pos'].append(positions)
        stats['all_mom'].append(momentums)
        stats['all_ham'].append(hamiltonians)
    plot_groups(stats, eps_values)
