"""
(c) December 2017 by Daniel Seita

By default, directories were saved in the directory:

    experiments/logs/nag-seed-1 

And each of these files looks like this:

[0] train-err:0.111780 train-nlik:0.469018 valid-err:0.109600 valid-nlik:0.465617 test-err:0.107300 test-nlik:0.438589
[1] train-err:0.073440 train-nlik:0.251816 valid-err:0.070900 valid-nlik:0.264946 test-err:0.074100 test-nlik:0.253118
[2] train-err:0.055940 train-nlik:0.189269 valid-err:0.059700 valid-nlik:0.212581 test-err:0.062600 test-nlik:0.206443
    ...
[997] train-err:0.001600 train-nlik:0.026220 valid-err:0.018200 valid-nlik:0.069288 test-err:0.016000 test-nlik:0.061597
[998] train-err:0.001580 train-nlik:0.026214 valid-err:0.018100 valid-nlik:0.069293 test-err:0.016000 test-nlik:0.061594
[999] train-err:0.001580 train-nlik:0.026206 valid-err:0.018100 valid-nlik:0.069297 test-err:0.016000 test-nlik:0.061588

So I parse that and collect everything into one figure. 

Update: I had to remove a few lines from 3 SGHMC results due to overflow errors.
But it should work. It's not a big deal, it was just once out of 1000 epochs.
"""

import argparse
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True)

# Some matplotlib settings.
plt.style.use('seaborn-darkgrid')
error_region_alpha = 0.25
LOGDIR = 'experiments/logs/'
FIGDIR = 'experiments/figures/'
title_size = 22
tick_size = 17
legend_size = 17
ysize = 18
xsize = 18
lw = 3
ms = 8
colors = ['red', 'blue', 'yellow', 'black']


def parse(directories):
    """ Parse line based on the pattern we know (see comments at the top). """
    errors_v = []
    errors_t = []
    neglogliks_v = []
    neglogliks_t = []

    for dd in directories:
        with open(LOGDIR+dd, 'r') as f:
            all_lines = [x.strip('\n').split(':') for x in f.readlines()]
        
        # Look at indices 3, 4, 5, 6, since we know the exact form.
        for idx in range(len(all_lines)):
            new_list = (all_lines[idx])[3:]
            #print(new_list)
            new_list[0] = float( (new_list[0].split(' '))[0] )
            new_list[1] = float( (new_list[1].split(' '))[0] )
            new_list[2] = float( (new_list[2].split(' '))[0] )
            new_list[3] = float( new_list[3] )
            #print(new_list)
            all_lines[idx] = new_list

        results = np.array(all_lines)
        assert results.shape == (1000,4)

        # Add to our lists.
        errors_v.append(results[:,0])
        errors_t.append(results[:,1])
        neglogliks_v.append(results[:,2])
        neglogliks_t.append(results[:,3])

    errors_v = np.array(errors_v)
    errors_t = np.array(errors_t)
    neglogliks_v = np.array(neglogliks_v)
    neglogliks_t = np.array(neglogliks_t)
    print("errors_v.shape: {}".format(errors_v.shape))
    return errors_v, errors_t, neglogliks_v, neglogliks_t


def axarr_plot(axarr, row, col, xcoords, mean, std, cc, name):
    axarr[row,col].plot(xcoords, mean, lw=lw, color=cc, label=name)
    axarr[row,col].fill_between(xcoords, mean-std, mean+std, 
            alpha=error_region_alpha, facecolor=cc)


def plot(dirs):
    plot_names = {
        'nag':   [d for d in dirs if 'nag-seed' in d],
        'sgd':   [d for d in dirs if 'sgd-seed' in d],
        'sgld':  [d for d in dirs if 'sgld-seed' in d],
        'sghmc': [d for d in dirs if 'sghmc-seed' in d]
    }

    # Top row: validation error and then neg log-like.
    # Bottom row: test error and then neg log-like.
    fig,axarr = plt.subplots(2, 2, figsize=(20,16))

    for cc,name in zip(colors,plot_names):
        directories = plot_names[name]
        errors_v, errors_t, neglogliks_v, neglogliks_t = parse(directories)
        assert errors_v.shape[0] < errors_v.shape[1]

        mean_err_v = np.mean(errors_v, axis=0)
        mean_lik_v = np.mean(neglogliks_v, axis=0)
        std_err_v  = np.std(errors_v, axis=0)
        std_lik_v  = np.std(neglogliks_v, axis=0)

        mean_err_t = np.mean(errors_t, axis=0)
        mean_lik_t = np.mean(neglogliks_t, axis=0)
        std_err_t  = np.std(errors_t, axis=0)
        std_lik_t  = np.std(neglogliks_t, axis=0)

        xcoords = np.arange(len(mean_err_v))
        axarr_plot(axarr, 0, 0, xcoords, mean_err_v, std_err_v, cc, name)
        axarr_plot(axarr, 0, 1, xcoords, mean_lik_v, std_lik_v, cc, name)
        axarr_plot(axarr, 1, 0, xcoords, mean_err_t, std_err_t, cc, name)
        axarr_plot(axarr, 1, 1, xcoords, mean_lik_t, std_lik_t, cc, name)

    # Bells and whistles
    for row in range(2):
        for col in range(2):
            axarr[row,col].set_xlabel("Epochs (# MCMC Samples)", fontsize=xsize)
            axarr[row,col].tick_params(axis='x', labelsize=tick_size)
            axarr[row,col].tick_params(axis='y', labelsize=tick_size)
            axarr[row,col].legend(loc="best", prop={'size':legend_size})
            if row == 0:
                axarr[row,col].set_ylabel("Classification Error", fontsize=ysize)
                axarr[row,col].set_ylim([0.015, 0.06])
            else:
                axarr[row,col].set_ylabel("Negative Log Likelihood", fontsize=ysize)
                axarr[row,col].set_ylim([0, 0.3])
    axarr[0,0].set_title("Valid Set Error", fontsize=title_size)
    axarr[0,1].set_title("Valid Set NegLogLik", fontsize=title_size)
    axarr[1,0].set_title("Test Set Error", fontsize=title_size)
    axarr[1,1].set_title("Test Set NegLogLik", fontsize=title_size)
    plt.tight_layout()
    plt.savefig(FIGDIR+"verify_sghmc_results.png")


if __name__ == "__main__":
    dirs = sorted([e for e in os.listdir(LOGDIR) if 'seed' in e])
    print("Plotting one figure for the directories: {}".format(dirs))
    plot(dirs)
