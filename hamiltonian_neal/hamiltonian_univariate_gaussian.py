"""
The simplest example in Neal's chapter. Throughout, we define a state with
(q,p), so q comes first, then p. A couple of observations:

1. There's no need to even define q(t) and p(t). The methods simply take our
previously computed stuff for the next iteration ... and knowing q(t) and p(t)
exactly would be cheating anyway.

2. Be careful with numpy stuff, sometimes it's [x] sometimes [[x]], sometimes x,
etc. That caused me some confusion. ANOTHER thing that was confusing was I
forgot to set the type of the point to be float ... yeah. =(

3. Indeed, it looks like Radford Neal meant to make the p updates before the q
updates, though he says "similar performance" can be achieved with the reverse.

Just to remind myself, this should really be a bunch of first-order Taylor
series expansions. We have one term plus another term for the first-order
derivative.
"""

import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import sys
FIG_DIR = "draft_figures/"


def _dqdt(s):
    """ Dynamics with (d/dt)q(t) = (d/dp)(p^2)/2 = p. """
    return s[1]


def _dpdt(s):
    """ Dynamics with (d/dt)p(t) = -(d/dq)(q^2)/2 = -q. """
    return -s[0]


def run_euler(start, eps, num_steps):
    """ Runs HMC with Euler's method. No vectorization =(. """
    t = 0
    points = np.array(start.reshape(2,1))
    curr_pt = start
    for _ in range(num_steps):
        t += eps
        p_next = curr_pt[1] + eps * _dpdt(curr_pt)
        q_next = curr_pt[0] + eps * _dqdt(curr_pt)
        curr_pt = np.array([q_next,p_next])
        points = np.hstack((points, curr_pt.reshape(2,1)))
    print("Euler's method, points.shape = {}.".format(points.shape))
    return points


def run_improved_euler(start, eps, num_steps):
    """ 
    Runs HMC with the improved Euler's method. Now we can set curr_pt[1] (the
    position) first, before calling the position update.
    """
    points = np.array(start.reshape(2,1))
    curr_pt = start
    for _ in range(num_steps):
        curr_pt[1] = curr_pt[1] + eps * _dpdt(curr_pt)
        curr_pt[0] = curr_pt[0] + eps * _dqdt(curr_pt)
        points = np.hstack((points, curr_pt.reshape(2,1)))
    print("Improved Euler's method, points.shape = {}.".format(points.shape))
    return points


def run_leapfrog(start, eps, num_steps):
    """ Runs HMC with the Leapfrog method. """
    points = np.array(start.reshape(2,1))
    curr_pt = start
    for _ in range(num_steps):
        curr_pt[1] = curr_pt[1] + (eps/2) * _dpdt(curr_pt)
        curr_pt[0] = curr_pt[0] + eps * _dqdt(curr_pt)
        curr_pt[1] = curr_pt[1] + (eps/2) * _dpdt(curr_pt)
        points = np.hstack((points, curr_pt.reshape(2,1)))
    print("Leapfrog method, points.shape = {}.".format(points.shape))
    return points


def plot(euler_pts, im_euler_pts, leapfrog_pts1, leapfrog_pts2):
    """ Plotting!! """
    fig, axarr = plt.subplots(2,2, figsize=(10,9.5))

    axarr[0,0].plot(euler_pts[0], euler_pts[1], '-bo')
    axarr[0,0].set_title("Euler's Method")
    axarr[0,1].plot(im_euler_pts[0], im_euler_pts[1], '-bo')
    axarr[0,1].set_title("Improved Euler's Method")
    axarr[1,0].plot(leapfrog_pts1[0], leapfrog_pts1[1], '-bo')
    axarr[1,0].set_title("Leapfrog Method (eps=0.3)")
    axarr[1,1].plot(leapfrog_pts2[0], leapfrog_pts2[1], '-bo')
    axarr[1,1].set_title("Leapfrog Method (eps=1.2)")

    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x,y)
    F = X**2 + Y**2 - 1.0
    for i in range(2):
        for j in range(2):
            axarr[i,j].set_xlabel("Position (q)")
            axarr[i,j].set_ylabel("Momentum (p)")
            axarr[i,j].set_xlim([-2.5, 2.5])
            axarr[i,j].set_ylim([-2.5, 2.5])
            axarr[i,j].contour(X,Y,F,[0], colors='k')
    plt.tight_layout()
    plt.savefig(FIG_DIR+"univariate_gaussians.png")


if __name__ == "__main__":
    s = np.array([0,1]).astype('float32')
    euler_pts     = run_euler(start=np.copy(s), eps=0.3, num_steps=20)
    im_euler_pts  = run_improved_euler(start=np.copy(s), eps=0.3, num_steps=20)
    leapfrog_pts1 = run_leapfrog(start=np.copy(s), eps=0.3, num_steps=20)
    leapfrog_pts2 = run_leapfrog(start=np.copy(s), eps=1.2, num_steps=20)
    plot(euler_pts, im_euler_pts, leapfrog_pts1, leapfrog_pts2)
