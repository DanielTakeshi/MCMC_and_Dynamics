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


def f_U(q, cov):
    """ The U term within the Hamiltonian function. """
    return 0.5 * (q.T).dot(np.linalg.inv(cov)).dot(q)


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
    p = np.random.normal(size=q.shape)
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
    print(test_stat)
    if (np.random.rand() < test_stat):
        return (q, 1, proposed_H[0]) # accept proposed sample
    else:
        return (current_q, 0, current_H[0])
   

def run_samples(N, eps, L):
    """
    Runs the example in Setion 5.3.3.2 in Neal's paper. I am assuming he used
    the same starting q?
    """
    cov = np.array([[1, 0.98],[0.98, 1]]).astype('float32')
    q = np.array([[-1.50],[-1.55]]).astype('float32')
    positions = np.zeros((2,N))
    hamiltonians = []
    acc_rate = 0.0
    
    for i in range(N):
        q, a, h = leapfrog(q, cov, eps, L)
        positions[:,i] = q.reshape(2,)
        acc_rate += a
        hamiltonians.append(h)

    hamiltonians = np.array(hamiltonians)
    acc_rate = acc_rate / N
    return positions, hamiltonians, acc_rate


def plot(positions, hamiltonians, acc_rate):
    """ 
    This creates two plots. One is to match Neal's figure. The other is for
    understanding the gradients of the momentums. 
    """
    fig, axarr = plt.subplots(1,3, figsize=(15,4.5))
    axarr[0].set_title("Random-walk Metropolis", fontsize=24)
    axarr[1].plot(positions[0,:], positions[1,:], '-ro')
    axarr[1].set_title("Hamiltonian Monte Carlo", fontsize=24)
    axarr[1].set_xlim([-2.5,2.5])
    axarr[1].set_ylim([-2.5,2.5])
    axarr[2].plot(hamiltonians, '-ro')
    axarr[2].set_title("Hamiltonian (acc-rate={:.3f})".format(acc_rate), fontsize=24)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"bivariate_gaussians_many_samples.png")

if __name__ == "__main__":
    positions, hamiltonians, acc_rate = run_samples(N=20, eps=0.18, L=20)
    plot(positions, hamiltonians, acc_rate)
