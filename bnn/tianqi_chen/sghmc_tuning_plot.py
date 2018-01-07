"""
Author: Daniel Seita

Use this for investigating SGHMC results (I did this from Tianqi's code base
directly, not my modification). Files look like this:

[0] train-err:0.111780 train-nlik:0.469018 valid-err:0.109600 valid-nlik:0.465617 test-err:0.107300 test-nlik:0.438589
[1] train-err:0.073440 train-nlik:0.251816 valid-err:0.070900 valid-nlik:0.264946 test-err:0.074100 test-nlik:0.253118
[2] train-err:0.055940 train-nlik:0.189269 valid-err:0.059700 valid-nlik:0.212581 test-err:0.062600 test-nlik:0.206443
    ...
[997] train-err:0.001600 train-nlik:0.026220 valid-err:0.018200 valid-nlik:0.069288 test-err:0.016000 test-nlik:0.061597
[998] train-err:0.001580 train-nlik:0.026214 valid-err:0.018100 valid-nlik:0.069293 test-err:0.016000 test-nlik:0.061594
[999] train-err:0.001580 train-nlik:0.026206 valid-err:0.018100 valid-nlik:0.069297 test-err:0.016000 test-nlik:0.061588

So I parse that and collect everything into one figure.
"""

import argparse
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

COLORS = ['red', 'blue', 'yellow', 'gold', 'purple', 'orange', 'darkblue',
          'cyan', 'pink', 'brown', 'green', 'silver']
SEEDS  = 12
EPOCHS = 800
BURNIN = 50


def parse(directories):
    """ Parse line based on the pattern we know (see comments at the top).

    The directories here should be all the same, EXCEPT for the seeds.
    """
    errors_v = []
    errors_t = []
    neglogliks_v = []
    neglogliks_t = []

    for dd in directories:
        with open('logs/sghmc-tune/'+dd, 'r') as f:
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
        assert results.shape == (EPOCHS,4)

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


def plot(dirs):
    """ Left: seed 0 only. Right: all seeds, zoomed-in. """
    nrows = 1
    ncols = 2
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10*ncols,8*nrows))

    errors_v, errors_t, neglogliks_v, neglogliks_t = parse(dirs)
    assert len(dirs) == SEEDS
    assert len(errors_v) == len(errors_t) == len(neglogliks_v) == len(neglogliks_t)
    assert errors_v.shape[0] < errors_v.shape[1]
    assert errors_v.shape == errors_t.shape == (SEEDS, EPOCHS)
    assert np.max(errors_v) < 1.0 and np.max(errors_t) < 1.0
    assert np.min(errors_v) > 0.0 and np.min(errors_t) > 0.0
    assert np.min(neglogliks_v) > 0.0 and np.min(neglogliks_t) > 0.0

    # Don't forget to apply the burn-in for future plotting!
    xcoords = np.arange(len(errors_v[0]))[BURNIN:]

    # Handle all other random seeds ... note that 10 is a bit out of order since
    # it actually appears after 1 in the ABCs, but that's a really minor thing.
    for ss in range(1,SEEDS):
        axarr[1].plot(
            xcoords, errors_t[ss,BURNIN:], lw=1, color=COLORS[ss],
            label='seed-{}-bestval-{:.5f}'.format(ss, np.min(errors_t[ss,BURNIN:]))
        )

    for col in range(ncols):
        axarr[col].plot(
            xcoords, errors_t[0,BURNIN:], lw=4, color='black',
            label='seed-0-bestval-{:.5f}'.format(np.min(errors_t[0,BURNIN:]))
        )
        axarr[col].set_xlabel("Epochs (# MCMC Samples)", fontsize=xsize)
        axarr[col].tick_params(axis='x', labelsize=tick_size)
        axarr[col].tick_params(axis='y', labelsize=tick_size)
        axarr[col].legend(loc="best", prop={'size':legend_size})
        axarr[col].set_ylabel("Test Error", fontsize=ysize)
        axarr[col].set_xlim([0, EPOCHS])
        axarr[col].set_ylim([0.015, 0.05])

    axarr[0].set_title("Test Error, Seed 0 Only", fontsize=title_size)
    axarr[1].set_title("Test Error, {} Random Seeds".format(SEEDS), fontsize=title_size)

    plt.tight_layout()
    plt.savefig("figures/sghmc_from_tianqi.png")


if __name__ == "__main__":
    dirs = sorted([e for e in os.listdir('logs/sghmc-tune/') if 'seed' in e])
    print("Plotting one figure for the directories: {}".format(dirs))
    plot(dirs)
