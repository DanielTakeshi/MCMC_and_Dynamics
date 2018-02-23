# To Compare SGHMC and MOMSGD

Change directory into `experiments`. Use the commands:

- `./benchmark_sghmc.sh`
- `./benchmark_momsgd.sh`

These go for 50 random seeds, so just be careful! Also, these assume that we
have investigated and found "almost optimal" hyper-parameters for the two
algorithms.

Run `python plot_momsgd_vs_sghmc.py` to generate a figure.
