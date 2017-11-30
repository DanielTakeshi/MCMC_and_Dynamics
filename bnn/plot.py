"""
Do `python plot.py logs/mnist/` or something like that.
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import sys
from os.path import join
from pylab import subplots
plt.style.use('seaborn-darkgrid')
sns.set_context(rc={'lines.markeredgewidth': 1.0})
np.set_printoptions(edgeitems=180, linewidth=180, suppress=True)


# Some matplotlib settings.
FIGDIR = 'figures/'
title_size = 22
tick_size = 18
legend_size = 17
ysize = 18
xsize = 18
lw = 2
ms = 8
error_region_alpha = 0.3

# Attributes to include in a plot.
ATTRIBUTES = ["ValidAcc", "HMCAcceptRateEpoch", "ValidLoss", 
        "HamiltonianOldMean", "HamiltonianNewMean"]
COLORS = ['red', 'blue', 'yellow', 'black', 'orange']
DIR_TO_LABEL = {'seed_43': 'mb=500, L=3, eps=0.000l, T=1', 
                'seed_44': 'mb=500, L=3, eps=0.001,  T=10',
                'seed_45': 'mb=500, L=3, eps=0.01,   T=100'}


def plot_one_directory(args, dirnames, figname):
    """ 
    Here, `dirname` contains directories named by the random seed.  Then inside
    each of those seeds, there's a `log.txt` file.
    """
    logdir = args.logdir
    num = int((len(ATTRIBUTES)+1) / 2)
    fig, axes = subplots(nrows=num, ncols=2, figsize=(14,4*num))

    for (dd, cc) in zip(dirnames, COLORS):
        A = np.genfromtxt(join(logdir, dd, 'log.txt'), 
                          delimiter='\t', 
                          dtype=None, 
                          names=True)
        x = A['Epochs']

        for row in range(num):
            for col in range(2):
                idx = 2*row + col
                if idx >= len(ATTRIBUTES):
                    break
                attr = ATTRIBUTES[idx]
                axes[row,col].plot(x, A[attr], '-', lw=lw, color=cc,
                                   label=DIR_TO_LABEL[dd])
                axes[row,col].legend(loc='best', ncol=1, prop={'size':legend_size})

    # Bells and whistles.
    for row in range(num):
        for col in range(2):
            idx = 2*row + col
            if idx >= len(ATTRIBUTES):
                break
            axes[row,col].set_title(ATTRIBUTES[idx], fontsize=title_size)
            axes[row,col].tick_params(axis='x', labelsize=tick_size)
            axes[row,col].tick_params(axis='y', labelsize=tick_size)
            axes[row,col].set_xlabel("Training Epochs", fontsize=xsize)
            axes[row,col].set_ylabel(ATTRIBUTES[idx], fontsize=ysize)
    axes[0,0].set_ylim([0.0, 1.0]) # Accuracy
    axes[0,1].set_ylim([0.0, 0.5]) # HMC Accept Rate

    plt.tight_layout()
    plt.savefig(figname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', help="Example, logs/mnist/")
    args = parser.parse_args()
    assert args.logdir[-1] == '/'

    dirnames = sorted(os.listdir(args.logdir))
    figname = FIGDIR+args.logdir[:-1]+'.png' # Get rid of trailing slash.
    figname = figname.replace('logs/', '')

    print("plotting to: {}\nwith these seeds: {}".format(figname, dirnames))
    plot_one_directory(args, dirnames, figname)
