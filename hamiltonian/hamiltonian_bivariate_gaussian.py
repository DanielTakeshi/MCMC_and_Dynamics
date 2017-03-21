"""
Simple Hamiltonian. Based on Section 5.3.3.1 in Radford Neal's article, 2010.

Notation:

    q = position
    p = momentum

Yeah, sorry I don't know why it's like that. Seems un-necessarily confusing.
"""

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import sys


def f_H(q, p, cov):
    """ The Hamiltonian function. """
    return 0.5 * ((q.T).dot(np.linalg.inv(cov)).dot(q) + (p.T).dot(p))


def f_U(q, cov):
    """ The U term """
    return 0.5 * (q.T).dot(np.linalg.inv(cov)).dot(q)


def grad_U(q, cov):
    """ The grad_U term. """
    return (np.linalg.inv(cov)).dot(q)


def leapfrog(current_q, cov, eps=0.25, L=25):
    """ 
    Runs the leapfrog method. This follows Neal's pseudocode in his paper, which
    seems misleading because it's only for *one* position variable. 
    """ 
    q = current_q 
    p = np.random.normal(size=q.shape)
    current_p = p

    # Half step for momentum
    p = p - eps * grad_U(q, cov) / 2.

    for i in range(L):
        # Make a full step for the position
        q = q + eps * p
        # Make a full step for the momentum, except at end of trajectory
        if (i != L):
            p = p - eps * grad_U(q, cov)

    # Make a half step for momentum at the end, and negate (still confused).
    p = p - eps * grad_U(q, cov) / 2.
    p = -p

    current_U = f_U(current_q, cov)
    current_K = sum(current_p**2) / 2.
    proposed_U = f_U(q, cov)
    proposed_K = sum(p**2) / 2.

    if (np.random.rand() < np.exp(current_U-proposed_U+current_K-proposed_K)):
        return q, p  # accept
    else:
        return current_q, p  # reject


def run(N=50):
    """
    Runs the example in Setion 5.3.3.1 in Neal 2010, using the provided initial
    states, which are column vectors.
    """
    cov = np.array([[1, 0.95],[0.95, 1]])
    q = np.array([[-1.50],[-1.55]])
    p = np.array([[-1],[1]]) # This is only for the _initial_ state.
    U = f_H(q, p, cov)
    positions = [q]
    momentums = [p]
    hamiltonians = [U]

    # Call the leapfrog method N times. Hopefully grad_U is computed this way.
    for i in range(N-1):
        p, q = leapfrog(q, cov)
        H = f_H(q, p, cov)
        positions.append(q)
        momentums.append(p)
        hamiltonians.append(H)

    positions = np.array(positions).reshape(N,2)
    momentums = np.array(momentums).reshape(N,2)
    hamiltonians = np.array(hamiltonians).reshape(N,1)
    return positions, momentums, hamiltonians


def plot(positions, momentums, hamiltonians, figdir="draft_figures/"):
    """ 
    Plots the positions, momentums, and hamiltonian value, in three separate
    plots.  
    """
    fig, axarr = plt.subplots(1,3, figsize=(15, 4))

    axarr[0].scatter(positions[:,0], positions[:,1])
    axarr[0].set_title("Positions")

    axarr[1].scatter(momentums[:,0], momentums[:,1])
    axarr[1].set_title("Momentums")
    
    axarr[2].plot(hamiltonians)
    axarr[2].set_title("Value of Hamiltonian")
    
    plt.tight_layout()
    plt.savefig(figdir+"simple_gaussians.png")


if __name__ == "__main__":
    positions, momentums, hamiltonians = run()
    plot(positions, momentums, hamiltonians)
