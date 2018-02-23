"""
Benchmarks SGHMC versus MOM+SGD. I saved log files looking like this:

[0] train-err:0.111780 train-nlik:0.469018 valid-err:0.109600 valid-nlik:0.465617 test-err:0.107300 test-nlik:0.438589
[1] train-err:0.073440 train-nlik:0.251816 valid-err:0.070900 valid-nlik:0.264946 test-err:0.074100 test-nlik:0.253118
[2] train-err:0.055940 train-nlik:0.189269 valid-err:0.059700 valid-nlik:0.212581 test-err:0.062600 test-nlik:0.206443
    ...
[997] train-err:0.001600 train-nlik:0.026220 valid-err:0.018200 valid-nlik:0.069288 test-err:0.016000 test-nlik:0.061597
[998] train-err:0.001580 train-nlik:0.026214 valid-err:0.018100 valid-nlik:0.069293 test-err:0.016000 test-nlik:0.061594
[999] train-err:0.001580 train-nlik:0.026206 valid-err:0.018100 valid-nlik:0.069297 test-err:0.016000 test-nlik:0.061588

So I parse that and collect everything into one figure. 

(c) February 2018 by Daniel Seita
"""
import argparse, matplotlib, os, pickle, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True)

# CHANGE DIRECTORIES AS NEEDED
FIGDIR  = 'experiments/figures/'
MOMSGD  = 'experiments/logs-momsgd/'
SGHMC   = 'experiments/logs-sghmc/'
VERSION = 1
R_SEEDS = 50

# Some matplotlib settings.
plt.style.use('seaborn-darkgrid')
error_region_alpha = 0.25
title_size = 26
tick_size = 21
legend_size = 21
xsize, ysize = 21, 21
lw, ms = 3, 8
colors = ['red', 'blue', 'yellow', 'black']


def parse(directories, head_dir):
    """ Parse line based on the pattern we know (see comments at the top). """
    errors_v = []
    errors_t = []
    neglogliks_v = []
    neglogliks_t = []

    for dd in directories:
        print("Now on directory: {}".format(dd))
        with open(head_dir+dd, 'r') as f:
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
        neglogliks_v.append(results[:,1])
        errors_t.append(results[:,2])
        neglogliks_t.append(results[:,3])

    errors_v = np.array(errors_v)
    errors_t = np.array(errors_t)
    neglogliks_v = np.array(neglogliks_v)
    neglogliks_t = np.array(neglogliks_t)
    print("\nerrors_v.shape: {}".format(errors_v.shape))
    print("errors_t.shape: {}".format(errors_t.shape))
    print("neglogliks_v.shape: {}".format(neglogliks_v.shape))
    print("neglogliks_t.shape: {}".format(neglogliks_t.shape))
    return errors_v, errors_t, neglogliks_v, neglogliks_t


def axarr_plot(axarr, row, col, xcoords, mean, std, cc, name):
    axarr[row,col].plot(xcoords, mean, lw=lw, color=cc, label=name)
    axarr[row,col].fill_between(xcoords, mean-std, mean+std, 
            alpha=error_region_alpha, facecolor=cc)


def plot(momsgd_dirs, sghmc_dirs):
    """ Take a screenshot of the appropriate figure and put it in the paper.

    Top row: validation error and then neg log-like.
    Bottom row: test error and then neg log-like.
    """
    fig,axarr = plt.subplots(2, 2, figsize=(22,14))
    stuff_to_plot = [
        (momsgd_dirs, MOMSGD, "SGD+Mom, {} Trials".format(R_SEEDS)),
        (sghmc_dirs, SGHMC, "SGHMC, {} Trials".format(R_SEEDS)),
    ]

    # Iterate through MOMSGD, then SGHMC. 
    for cc,info in zip(colors,stuff_to_plot):
        dirs, head_dir, name = info
        print("\nNow on algorithm with dir {}, with {} logged dirs.\n".format(
                head_dir, len(dirs)))
        errors_v, errors_t, neglogliks_v, neglogliks_t = parse(dirs, head_dir)
        assert len(errors_v) == len(errors_t) == len(neglogliks_v) == len(neglogliks_t)
        assert errors_v.shape[0] < errors_v.shape[1]
        assert np.max(errors_v) < 1.0 and np.max(errors_t) < 1.0
        assert np.min(errors_v) > 0.0 and np.min(errors_t) > 0.0
        assert np.min(neglogliks_v) > 0.0 and np.min(neglogliks_t) > 0.0

        mean_err_v = np.mean(errors_v, axis=0)
        mean_lik_v = np.mean(neglogliks_v, axis=0)
        std_err_v  = np.std(errors_v, axis=0)
        std_lik_v  = np.std(neglogliks_v, axis=0)

        mean_err_t = np.mean(errors_t, axis=0)
        mean_lik_t = np.mean(neglogliks_t, axis=0)
        std_err_t  = np.std(errors_t, axis=0)
        std_lik_t  = np.std(neglogliks_t, axis=0)

        xcoords = np.arange(len(mean_err_v))
        assert len(mean_err_t) == len(xcoords) == 1000
        axarr_plot(axarr, 0, 0, xcoords, mean_err_v, std_err_v, cc, name)
        axarr_plot(axarr, 0, 1, xcoords, mean_lik_v, std_lik_v, cc, name)
        axarr_plot(axarr, 1, 0, xcoords, mean_err_t, std_err_t, cc, name)
        axarr_plot(axarr, 1, 1, xcoords, mean_lik_t, std_lik_t, cc, name)

    # Bells and whistles. Tweak these.
    for row in range(2):
        for col in range(2):
            axarr[row,col].set_xlabel("Epochs (# MCMC Samples)", fontsize=xsize)
            axarr[row,col].tick_params(axis='x', labelsize=tick_size)
            axarr[row,col].tick_params(axis='y', labelsize=tick_size)
            axarr[row,col].legend(loc="best", prop={'size':legend_size})
            if col == 0:
                axarr[row,col].set_ylabel("Classification Error", fontsize=ysize)
                axarr[row,col].set_ylim([0.014, 0.026])
            else:
                axarr[row,col].set_ylabel("Negative Log Likelihood", fontsize=ysize)
                axarr[row,col].set_ylim([0.040, 0.140])
    axarr[0,0].set_title("SGHMC vs SGD+Mom, Tuned H-Params, Validation Error", fontsize=title_size)
    axarr[0,1].set_title("SGHMC vs SGD+Mom, Tuned H-Params, Validation NegLogLik", fontsize=title_size)
    axarr[1,0].set_title("SGHMC vs SGD+Mom, Tuned H-Params, Test Error", fontsize=title_size)
    axarr[1,1].set_title("SGHMC vs SGD+Mom, Tuned H-Params, Test NegLogLik", fontsize=title_size)
    plt.tight_layout()
    savedir = FIGDIR+"sghmc_vs_momsgd_v"+str(VERSION).zfill(2)+".png"
    print("\nJust saved figure: {}".format(savedir))
    plt.savefig(savedir)


if __name__ == "__main__":
    momsgd = sorted([x for x in os.listdir(MOMSGD) if 'seed' in x])
    sghmc  = sorted([x for x in os.listdir(SGHMC) if 'seed' in x])
    plot(momsgd, sghmc)
