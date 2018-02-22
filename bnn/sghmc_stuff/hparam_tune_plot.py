"""
Author: Daniel Seita

Use this one by one for each of the four methods. This is only for
hyperparameter tuning.  Each file looks like this:

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
title_size = 22
tick_size = 17
legend_size = 17
ysize = 18
xsize = 18
lw = 3
ms = 8

COLORS = ['red', 'blue', 'yellow', 'black', 'purple', 'orange']
LOGDIR = 'experiments/logs/'
FIGDIR = 'experiments/figures/'
METHOD = 'sghmc' # NOTE this is what we adjust!
SEEDS  = 5


def parse(directories):
    """ Parse line based on the pattern we know (see comments at the top). 
    
    The directories here should be all the same, EXCEPT for the seeds.
    """
    errors_v = []
    errors_t = []
    neglogliks_v = []
    neglogliks_t = []

    for dd in directories:
        with open(LOGDIR+METHOD+'-tune/'+dd, 'r') as f:
            all_lines = [x.strip('\n').split(':') for x in f.readlines()]
        
        # Look at indices 3, 4, 5, 6, since we know the exact form.
        print(dd)
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
        neglogliks_v.append(results[:,1])
        errors_t.append(results[:,2])
        neglogliks_t.append(results[:,3])

    errors_v = np.array(errors_v)
    errors_t = np.array(errors_t)
    neglogliks_v = np.array(neglogliks_v)
    neglogliks_t = np.array(neglogliks_t)
    print("errors_v.shape: {}".format(errors_v.shape))
    print("errors_t.shape: {}".format(errors_t.shape))
    print("neglogliks_v.shape: {}".format(neglogliks_v.shape))
    print("neglogliks_t.shape: {}".format(neglogliks_t.shape))
    return errors_v, errors_t, neglogliks_v, neglogliks_t


def axarr_plot(axarr, row, col, xcoords, mean, std, cc, name):
    axarr[row,col].plot(xcoords, mean, lw=lw, color=cc, label=name)
    axarr[row,col].fill_between(xcoords, mean-std, mean+std, 
            alpha=error_region_alpha, facecolor=cc)


def plot_2x2(dirs):
    nrows = 2
    ncols = 2
    if METHOD == 'sgld':
        eta_terms = ['0.1', '0.4', '0.8', '1.0', '1.2', '1.4']
    elif METHOD == 'sghmc':
        eta_terms = ['0.01', '0.05', '0.08', '0.1', '0.2', '0.4']
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))

    for eta, cc in zip(eta_terms, COLORS):
        s1 = 'eta-{}-'.format(eta)
        directories = [d for d in dirs if s1 in d]
        print("\nPlotting with s1 = {}".format(s1))
        errors_v, errors_t, neglogliks_v, neglogliks_t = parse(directories)

        assert len(directories) == SEEDS
        assert len(errors_v) == len(errors_t) == len(neglogliks_v) == len(neglogliks_t)
        assert errors_v.shape[0] < errors_v.shape[1]
        assert np.max(errors_v) < 1.0 and np.max(errors_t) < 1.0
        assert np.min(errors_v) > 0.0 and np.min(errors_t) > 0.0
        assert np.min(neglogliks_v) > 0.0 and np.min(neglogliks_t) > 0.0

        # validation
        mean_err_v = np.mean(errors_v, axis=0)
        mean_lik_v = np.mean(neglogliks_v, axis=0)
        std_err_v  = np.std(errors_v, axis=0)
        std_lik_v  = np.std(neglogliks_v, axis=0)

        # test 
        mean_err_t = np.mean(errors_t, axis=0)
        mean_lik_t = np.mean(neglogliks_t, axis=0)
        std_err_t  = np.std(errors_t, axis=0)
        std_lik_t  = np.std(neglogliks_t, axis=0)

        xcoords = np.arange(len(mean_err_v))
        assert len(mean_err_t) == len(xcoords) == 1000
        axarr_plot(axarr, 0, 0, xcoords, mean_err_v, std_err_v, cc, 
                   s1+'bestval='+str(np.min(mean_err_v)))
        axarr_plot(axarr, 0, 1, xcoords, mean_lik_v, std_lik_v, cc,
                   s1+'bestval='+str(np.min(mean_lik_v)))
        axarr_plot(axarr, 1, 0, xcoords, mean_err_t, std_err_t, cc,
                   s1+'bestval='+str(np.min(mean_err_t)))
        axarr_plot(axarr, 1, 1, xcoords, mean_lik_t, std_lik_t, cc,
                   s1+'bestval='+str(np.min(mean_lik_t)))

    # Bells and whistles
    for row in range(nrows):
        for col in range(ncols):
            axarr[row,col].set_xlabel("Epochs (# MCMC Samples)", fontsize=xsize)
            axarr[row,col].tick_params(axis='x', labelsize=tick_size)
            axarr[row,col].tick_params(axis='y', labelsize=tick_size)
            axarr[row,col].legend(loc="best", prop={'size':legend_size})
            if col == 0:
                axarr[row,col].set_ylabel("Classification Error", fontsize=ysize)
                axarr[row,col].set_ylim([0.00, 0.15])
            else:
                axarr[row,col].set_ylabel("Negative Log Likelihood", fontsize=ysize)
                axarr[row,col].set_ylim([0, 2.0])
    axarr[0,0].set_title("Valid Error, {}".format(METHOD), fontsize=title_size)
    axarr[0,1].set_title("Valid NegLogLik, {}".format(METHOD), fontsize=title_size)
    axarr[1,0].set_title("Test Error, {}".format(METHOD), fontsize=title_size)
    axarr[1,1].set_title("Test NegLogLik, {}".format(METHOD), fontsize=title_size)

    plt.tight_layout()
    plt.savefig(FIGDIR+ "verify_sghmc_results_" +METHOD+ ".png")


def plot_8x2(dirs):
    """ Eight rows. 

    - 0, 1: valid error
    - 2, 3: valid neglik
    - 4, 5: test error
    - 6, 7: test neglik

    Iterate through weight decay terms first. Then iterate through eta. Varying
    etas are in the same sub-plots.
    """
    nrows = 8
    ncols = 2
    if METHOD == 'sgd':
        eta_terms = ['0.1', '0.4', '0.7', '1.0', '1.3']
        wd_terms  = ['0.0001']
    elif METHOD == 'momsgd':
        eta_terms = ['0.01', '0.1', '0.4', '0.5', '0.7', '1.0']
        wd_terms  = ['0.00001']
    wd_offsets = [(0,0), (0,1), (1,0), (1,1)]
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))

    for wd, wd_off in zip(wd_terms, wd_offsets):
        for eta, cc in zip(eta_terms, COLORS):
            s1 = 'wd-{}-'.format(wd)
            s2 = 'eta-{}-'.format(eta)
            directories = [d for d in dirs if s1 in d and s2 in d]
            print("\nPlotting with s1, s2 = {}, {}".format(s1,s2))
            errors_v, errors_t, neglogliks_v, neglogliks_t = parse(directories)

            assert len(directories) == SEEDS
            assert len(errors_v) == len(errors_t) == len(neglogliks_v) == len(neglogliks_t)
            assert errors_v.shape[0] < errors_v.shape[1]
            assert np.max(errors_v) < 1.0 and np.max(errors_t) < 1.0
            assert np.min(errors_v) > 0.0 and np.min(errors_t) > 0.0
            assert np.min(neglogliks_v) > 0.0 and np.min(neglogliks_t) > 0.0

            # validation
            mean_err_v = np.mean(errors_v, axis=0)
            mean_lik_v = np.mean(neglogliks_v, axis=0)
            std_err_v  = np.std(errors_v, axis=0)
            std_lik_v  = np.std(neglogliks_v, axis=0)

            # test 
            mean_err_t = np.mean(errors_t, axis=0)
            mean_lik_t = np.mean(neglogliks_t, axis=0)
            std_err_t  = np.std(errors_t, axis=0)
            std_lik_t  = np.std(neglogliks_t, axis=0)

            xcoords = np.arange(len(mean_err_v))
            assert len(mean_err_t) == len(xcoords) == 1000
            dx,dy = wd_off
            axarr_plot(axarr, 0+dx, 0+dy, xcoords, mean_err_v, std_err_v, cc, 
                       s2+'bestval='+str(np.min(mean_err_v)))
            axarr_plot(axarr, 2+dx, 0+dy, xcoords, mean_lik_v, std_lik_v, cc,
                       s2+'bestval='+str(np.min(mean_lik_v)))
            axarr_plot(axarr, 4+dx, 0+dy, xcoords, mean_err_t, std_err_t, cc,
                       s2+'bestval='+str(np.min(mean_err_t)))
            axarr_plot(axarr, 6+dx, 0+dy, xcoords, mean_lik_t, std_lik_t, cc,
                       s2+'bestval='+str(np.min(mean_lik_t)))

            # titles
            axarr[0+dx,0+dy].set_title(
                    "Valid Error, {}, wd {}".format(METHOD, wd), 
                    fontsize=title_size
            )
            axarr[2+dx,0+dy].set_title(
                    "Valid NegLogLik, {}, wd {}".format(METHOD, wd), 
                    fontsize=title_size
            )
            axarr[4+dx,0+dy].set_title(
                    "Test Error, {}, wd {}".format(METHOD, wd), 
                    fontsize=title_size
            )
            axarr[6+dx,0+dy].set_title(
                    "Test NegLogLik, {}, wd {}".format(METHOD, wd), 
                    fontsize=title_size
            )

    # Bells and whistles
    for row in range(nrows):
        for col in range(ncols):
            axarr[row,col].set_xlabel("Epochs (# MCMC Samples)", fontsize=xsize)
            axarr[row,col].tick_params(axis='x', labelsize=tick_size)
            axarr[row,col].tick_params(axis='y', labelsize=tick_size)
            axarr[row,col].legend(loc="best", prop={'size':legend_size})
            if row in [0,1,4,5]:
                axarr[row,col].set_ylabel("Classification Error", fontsize=ysize)
                axarr[row,col].set_ylim([0.00, 0.15])
            else:
                axarr[row,col].set_ylabel("Negative Log Likelihood", fontsize=ysize)
                axarr[row,col].set_ylim([0, 2.0])

    plt.tight_layout()
    plt.savefig(FIGDIR+ "verify_sghmc_results_" +METHOD+ ".png")


if __name__ == "__main__":
    dirs = sorted([e for e in os.listdir(LOGDIR+METHOD+'-tune') if 'seed' in e])
    print("Plotting one figure for the directories: {}".format(dirs))
    if METHOD == 'sgd' or METHOD == 'momsgd':
        plot_8x2(dirs)
    else:
        plot_2x2(dirs)
