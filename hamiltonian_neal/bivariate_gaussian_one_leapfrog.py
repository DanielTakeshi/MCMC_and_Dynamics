"""
Simple Hamiltonian. Based on Section 5.3.3.1 in Radford Neal's article, 2010.

Notation:

    q = position
    p = momentum

Yeah, sorry I don't know why it's like that. Seems un-necessarily confusing. But
anyway, I managed to get this to match what he did. I didn't read this carefully
enough and tried to implement multiple samples, but this example is only based
on ONE sample. I also added a bit to understand the gradients.
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


def run_one_sample(eps, L):
    """
    Runs the example in Setion 5.3.3.1 in Neal 2010, using the provided initial
    states, which are column vectors. Calls the leapfrog method ONE time, to
    generate ONE possible candidate point. This caused me some confusion.
    """
    cov = np.array([[1, 0.95],[0.95, 1]]).astype('float32')
    q = np.array([[-1.50],[-1.55]]).astype('float32')
    p = np.array([[-1],[1]]).astype('float32')
    H = f_H(q, p, cov)
    positions = np.array(q)
    momentums = np.array(p)
    hamiltonians = [H[0]]

    # Now do the leapfrog stuff.
    p = p - (0.5*eps) * grad_U(q, cov) 

    for i in range(L):
        q = q + eps * p
        positions = np.hstack((positions, q))
        if (i != L):
            p = p - eps * grad_U(q, cov)
            momentums = np.hstack((momentums, p))
            hamiltonians.append(f_H(q, p, cov))

    p = p - (0.5*eps) * grad_U(q, cov)
    momentums = np.hstack((momentums, p))
    hamiltonians.append(f_H(q, p, cov))
    p = -p

    hamiltonians = np.array(hamiltonians)
    return positions, momentums, hamiltonians


def plot(positions, momentums, hamiltonians, figdir="draft_figures/"):
    """  Plotting!!! """
    fig, axarr = plt.subplots(1,3, figsize=(15,4.5))

    axarr[0].plot(positions[0,:], positions[1,:], '-ro')
    axarr[0].set_title("Position Coordinates")
    axarr[0].set_xlim([-2.5,2.5])
    axarr[0].set_ylim([-2.5,2.5])
    axarr[1].plot(momentums[0,:], momentums[1,:], '-ro')
    axarr[1].set_title("Momentum Coordinates")
    axarr[1].set_xlim([-2.5,2.5])
    axarr[1].set_ylim([-2.5,2.5])
    axarr[2].plot(hamiltonians, '-ro')
    axarr[2].set_title("Value of Hamiltonian)")

    plt.tight_layout()
    plt.savefig(FIG_DIR+"bivariate_gaussians_one_sample.png")


if __name__ == "__main__":
    positions, momentums, hamiltonians = run_one_sample(eps=0.25, L=24)
    plot(positions, momentums, hamiltonians)
