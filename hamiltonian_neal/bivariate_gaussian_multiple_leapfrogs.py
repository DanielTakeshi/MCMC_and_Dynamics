"""
Simple Hamiltonian. Based on Section 5.3.3.2 in Radford Neal's article, 2010.
This time, we actually sample multiple times (each single sample needs its own
sequence of leapfrog steps). NOTE: at some point, I should switch away from
using hstack, because that's very inefficient for numpy.
"""

import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import sys
FIG_DIR = "draft_figures/"


def f_H(q, p, cov):
    """ The Hamiltonian function. See Section 5.3.3.1. """
    return 0.5 * ((q.T).dot(np.linalg.inv(cov)).dot(q) + (p.T).dot(p))


def grad_U(q, cov):
    """ The grad_U term. """
    return (np.linalg.inv(cov)).dot(q)


def leapfrog(current_q, cov, eps, L):
    """ 
    Runs one leapfrog step. For this, think of q and p as the proposals, and the
    current_{q,p} stuff as the old stuff. Sorry if this is confusing. TODO I
    better check this because the acceptance rates look off.
    """
    q = np.copy(current_q)
    p = np.random.multivariate_normal(mean=np.array([0,0]), cov=np.eye(2)).reshape((2,1)) 
    current_p = np.copy(p)

    p = p - (0.5*eps) * grad_U(q, cov) 
    for i in range(L):
        q = q + eps * p
        if (i != L):
            p = p - eps * grad_U(q, cov)
    p = p - (0.5*eps) * grad_U(q, cov)
    p = -p

    current_H = f_H(current_q, current_p, cov)
    proposed_H = f_H(q, p, cov)
    test_stat = np.exp(-proposed_H + current_H)
    if (np.random.rand() < test_stat[0,0]):
        return (q, 1, proposed_H[0]) # accept proposed sample
    else:
        return (current_q, 0, current_H[0])


def run_samples_hmc(s, covariance, N, eps, L):
    """
    Runs the example in Setion 5.3.3.2 in Neal's paper using Hamiltonian Monte
    Carlo. I am assuming he used the same starting q?
    """
    positions = np.zeros((2,N))
    acc_rate = 0.0
    hamiltonians = []
    q = np.copy(s)
    
    for i in range(N):
        q, a, h = leapfrog(q, covariance, eps, L)
        positions[:,i] = q.reshape(2,)
        acc_rate += a
        hamiltonians.append(h)

    hamiltonians = np.array(hamiltonians)
    acc_rate = acc_rate / N
    return positions, acc_rate, hamiltonians


def run_samples_rw(s, cov_rw, cov_distr, N):
    """ 
    Runs a simple random walk, with provided covariance.  Here, the desired
    distribution is proprtional to exp(-0.5*q^T*inv(cov_distr)*q) so this is the
    value p(q) used in the ratio p(q')/p(q) for the acceptance test:

    p(q')/p(q) = exp(-0.5*(q')^T*inv(cov_distr)*(q')) / exp(-0.5*q^T*inv(cov_distr)*q)
               = exp( -0.5*(q')^T*inv(cov_distr)*(q') + 0.5*q^T*inv(cov_distr)*q )

    I'm pretty sure this is right! The results look good ...
    """
    positions = np.zeros((2,N))
    acc_rate = 0.0
    curr_q = s
    cov_inv = np.linalg.inv(cov_distr)

    for i in range(N):
        prop_q = np.random.multivariate_normal(mean=curr_q.reshape(2,), cov=cov_rw).reshape((2,1))
        term_prop = 0.5 * (prop_q.T).dot(cov_inv).dot(prop_q)
        term_curr = 0.5 * (curr_q.T).dot(cov_inv).dot(curr_q)
        #test_stat = np.exp((-term_prop + term_curr)[0,0])
        test_stat = (-term_prop + term_curr)[0,0] # More numerically stable
        if (np.log(np.random.random()) < test_stat):
            curr_q = prop_q
            acc_rate += 1
        positions[:,i] = curr_q.reshape(2,)

    acc_rate = acc_rate / N
    return positions, acc_rate


def plot(positions, acc_rate, hamiltonians, positions_rw, acc_rate_rw):
    """  For plotting. """
    fig, axarr = plt.subplots(1,3, figsize=(15,4.5))

    axarr[0].plot(positions_rw[0,:][::20], positions_rw[1,:][::20], '-ro')
    axarr[0].set_title("Random-walk Metropolis ({:.3f})".format(acc_rate_rw), fontsize=20)
    axarr[0].set_xlim([-2.5,2.5])
    axarr[0].set_ylim([-2.5,2.5])

    axarr[1].plot(positions[0,:], positions[1,:], '-ro')
    axarr[1].set_title("Hamiltonian Monte Carlo ({:.3f})".format(acc_rate), fontsize=20)
    axarr[1].set_xlim([-2.5,2.5])
    axarr[1].set_ylim([-2.5,2.5])

    axarr[2].plot(hamiltonians, '-ro')
    axarr[2].set_title("Values of Hamiltonian", fontsize=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"bivariate_gaussians_many_samples.png")


if __name__ == "__main__":
    """ Define different covariances, but keep the same starting location. """
    cov_ham = np.array([[1, 0.98],[0.98, 1]]).astype('float32')
    cov_rw = np.array([[1, 0],[0, 1]]).astype('float32') * (0.18**2)
    start_q = np.array([[0],[0]]).astype('float32')

    positions, acc_rate, hamiltonians = run_samples_hmc(s=np.copy(start_q),
                                                        covariance=cov_ham,
                                                        N=20, 
                                                        eps=0.06, 
                                                        L=30)
    positions_rw, acc_rate_rw = run_samples_rw(s=np.copy(start_q),
                                               cov_rw=cov_rw,
                                               cov_distr=cov_ham,
                                               N=400)
    plot(positions, acc_rate, hamiltonians, positions_rw, acc_rate_rw)
