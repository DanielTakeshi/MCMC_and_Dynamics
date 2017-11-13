"""
The simplest example in Neal's chapter. Throughout, we define a state with
(q,p), so q comes first, then p. A couple of observations:

1. There's no need to even define q(t) and p(t). The methods simply take our
previously computed stuff for the next iteration ... and knowing q(t) and p(t)
exactly would be cheating anyway.

2. Be careful with numpy stuff, sometimes it's [x] sometimes [[x]], sometimes x,
etc. That caused me some confusion. ANOTHER thing that was confusing was I
forgot to set the type of the point to be float ... yeah. =( If you pass in
numpy arrays and you modify the elements, it still modifies the elements ...
it's passing a reference by value ... this is why I have a bunch of copies.

3. Indeed, it looks like Radford Neal meant to make the p updates before the q
updates (i.e., momentum updates before the position updates), though he says
"similar performance" can be achieved with the reverse.

Just to remind myself, this should really be a bunch of first-order Taylor
series expansions. We have one term plus another term for the first-order
derivative.
"""
import numpy as np
np.set_printoptions(suppress=True)
import sys

# Matplotlib stuff
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
FIG_DIR = "draft_figures/"
title_size = 18
xsize = 15
ysize = 15
legend_size = 15
tick_size = 15


def _dqdt(s):
    """ Dynamics with (d/dt)q(t) = (d/dp)(p^2)/2 = p. """
    return s[1]


def _dpdt(s):
    """ Dynamics with (d/dt)p(t) = -(d/dq)(q^2)/2 = -q. """
    return -s[0]


def run_euler(start, eps, num_steps):
    """ Runs HMC with Euler's method. No vectorization =(. """
    curr_pt = np.copy(start)
    points = [curr_pt]
    for _ in range(num_steps):
        p_next = curr_pt[1] + eps * _dpdt(curr_pt)
        q_next = curr_pt[0] + eps * _dqdt(curr_pt) # old curr_pt
        curr_pt = np.array([q_next,p_next])
        points.append(curr_pt)
    points = np.array(points)
    print("Euler's method, points.shape = {}.".format(points.shape))
    return points


def run_improved_euler(start, eps, num_steps):
    """ 
    Runs HMC with the improved Euler's method. Now we can set curr_pt[1] (the
    position) first, before calling the position update.
    """
    curr_pt = np.copy(start)
    points = [np.copy(curr_pt)]
    for _ in range(num_steps):
        curr_pt[1] = curr_pt[1] + eps * _dpdt(curr_pt)
        curr_pt[0] = curr_pt[0] + eps * _dqdt(curr_pt) # updated curr_pt
        points.append(np.copy(curr_pt))
    points = np.array(points)
    print("Improved Euler's method, points.shape = {}.".format(points.shape))
    return points


def run_leapfrog(start, eps, num_steps):
    """ Runs HMC with the Leapfrog method. """
    curr_pt = start
    points = [np.copy(curr_pt)]
    for _ in range(num_steps):
        curr_pt[1] = curr_pt[1] + (eps/2) * _dpdt(curr_pt)
        curr_pt[0] = curr_pt[0] + eps * _dqdt(curr_pt)
        curr_pt[1] = curr_pt[1] + (eps/2) * _dpdt(curr_pt)
        points.append(np.copy(curr_pt))
    points = np.array(points)
    print("Leapfrog method, points.shape = {}.".format(points.shape))
    return points


def plot(euler_pts, im_euler_pts, leapfrog_pts1, leapfrog_pts2):
    """ Plotting!! Points are np.arrays of shape (N, dimension). """
    fig, axarr = plt.subplots(2,2, figsize=(10,9.5))

    axarr[0,0].plot(euler_pts[:,0], euler_pts[:,1], '-bo')
    axarr[0,0].set_title("Euler's Method", fontsize=title_size)
    axarr[0,1].plot(im_euler_pts[:,0], im_euler_pts[:,1], '-bo')
    axarr[0,1].set_title("Improved Euler's Method", fontsize=title_size)
    axarr[1,0].plot(leapfrog_pts1[:,0], leapfrog_pts1[:,1], '-bo')
    axarr[1,0].set_title("Leapfrog Method (eps=0.3)", fontsize=title_size)
    axarr[1,1].plot(leapfrog_pts2[:,0], leapfrog_pts2[:,1], '-bo')
    axarr[1,1].set_title("Leapfrog Method (eps=1.2)", fontsize=title_size)

    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x,y)
    F = X**2 + Y**2 - 1.0
    for i in range(2):
        for j in range(2):
            axarr[i,j].set_xlabel("Position (q)", fontsize=xsize)
            axarr[i,j].set_ylabel("Momentum (p)", fontsize=ysize)
            axarr[i,j].set_xlim([-2.5, 2.5])
            axarr[i,j].set_ylim([-2.5, 2.5])
            axarr[i,j].contour(X,Y,F,[0], colors='k')
    plt.tight_layout()
    plt.savefig(FIG_DIR+"univariate_gaussians.png")


def plot_sets_of_points(epsilon_values, leapfrog_points):
    """ Plotting!! Points are np.arrays of shape (N, dimension). """
    ncols = 3
    nrows = int(len(leapfrog_points)/3)
    fig, axarr = plt.subplots(nrows, ncols, figsize=(5*nrows,5*ncols))

    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x,y)
    F = X**2 + Y**2 - 1.0

    for i in range(nrows):
        for j in range(ncols):
            ii = nrows*i + j
            eps = epsilon_values[ii]
            points = leapfrog_points[ii]
            axarr[i,j].plot(points[:,0], points[:,1], '-bo')
            axarr[i,j].set_title("Leapfrog, eps {:.2f}".format(eps), 
                                 fontsize=title_size)
            axarr[i,j].set_xlabel("Position (q)", fontsize=xsize)
            axarr[i,j].set_ylabel("Momentum (p)", fontsize=ysize)
            axarr[i,j].set_xlim([-2.5, 2.5])
            axarr[i,j].set_ylim([-2.5, 2.5])
            axarr[i,j].contour(X,Y,F,[0], colors='k')
            axarr[i,j].tick_params(axis='x', labelsize=tick_size)
            axarr[i,j].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(FIG_DIR+"univariate_gaussians_leapfrog_tests.png")


if __name__ == "__main__":
    s = np.array([0,1]).astype('float32')
    euler_pts     = run_euler(start=np.copy(s), eps=0.3, num_steps=20)
    im_euler_pts  = run_improved_euler(start=np.copy(s), eps=0.3, num_steps=20)
    leapfrog_pts1 = run_leapfrog(start=np.copy(s), eps=0.3, num_steps=20)
    leapfrog_pts2 = run_leapfrog(start=np.copy(s), eps=1.2, num_steps=20)
    plot(euler_pts, im_euler_pts, leapfrog_pts1, leapfrog_pts2)

    # Some additional investigation.
    epsilon_values = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 1.75, 2.0]
    leapfrog_points = []
    for eps in epsilon_values:
        points = run_leapfrog(start=np.copy(s), eps=eps, num_steps=40)
        leapfrog_points.append(points)
    plot_sets_of_points(epsilon_values, leapfrog_points)
