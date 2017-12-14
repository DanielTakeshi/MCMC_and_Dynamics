"""
Daniel: modified version of Tianqi's MNIST stuff so that I can test more
extensively.
"""

import argparse
import sys
sys.path.append('..')
import mnist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    assert (args.algo is not None) and (args.seed is not None)
    assert args.algo in ['sgd', 'momsgd', 'sgld', 'sghmc']

    # Get default prameters but later set what you want. This script has highest
    # priority, i.e. if we set things here they are actually going to be
    # applied and not overriden by something later.
    param = mnist.cfg_param()

    param.seed = args.seed
    param.batch_size = 500
    param.num_round  = 1000 # I changed from 800 -> 1000 
    param.num_hidden = 100

    # change the following line to PATH/TO/MNIST dataset
    # param.path_data = '../../../../../data/image/mnist/' # Tianqi's machine
    # param.path_data = '/home/daniel/mnist/' # My machine
    param.path_data = '/home/seita/MCMC_and_Dynamics/data/' # Triton

    # network type
    param.net_type = 'mlp2'
    # updating method
    param.updater  = args.algo
    # hyper parameter sampling: use gibbs for each parameter group
    param.hyperupdater = 'gibbs-sep'
    # number of burn-in round, start averaging after num_burn round
    param.num_burn = 50

    # learning rate, actually this is \gamma for SGHMC as we later divide by
    # 50k. I think this is the same as \epsilon in the paper for SGD/SGLD
    param.eta = args.eta
    param.wd  = args.wd

    # alpha, momentum decay. This should be fixed for the momentum methods.
    if args.algo == 'sgd' or args.algo == 'sgld':
        param.mdecay = 1.00
    elif args.algo == 'momsgd' or args.algo == 'sghmc':
        param.mdecay = 0.01
    
    # run the experiment
    mnist.run_exp( param )
