import argparse
import logz
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from nnet import Net
import utils as U


if __name__ == "__main__":
    pp = argparse.ArgumentParser()

    # General settings (TODO: hmc, sgd, adam, etc.)
    pp.add_argument('--algo', type=str, default='sgd',
                    help='See main method for details')
    pp.add_argument('--bsize', type=int, default=500,
                    help='batch size for HMC, really SGHMC...')
    pp.add_argument('--epochs', type=int, default=1000,
                    help='batch size for HMC, really SGHMC...')
    pp.add_argument('--seed', type=int, default=42,
                    help='random seed')
    pp.add_argument('--wdecay', type=float, default=0.0002,
                    help='weight decay for regularization')

    # HMC-specific
    pp.add_argument('--leapfrog_step', type=float, default=0.3,
                    help='step size for leapfrog method')
    pp.add_argument('--lrate_hmc', type=float, default=0.01,
                    help='step size for HMCX-based methods')
    pp.add_argument('--num_leapfrog', type=int, default=5,
                    help='number of leapfrog steps')

    # SGD-specific
    pp.add_argument('--lrate_sgd', type=float, default=0.01,
                    help='step size for SGD-based methods')
    pp.add_argument('--momentum', type=float, default=0.01,
                    help='if I need to use momentum somehow')

    # Bells and whistles
    pp.add_argument('--data_dir', type=str, 
                    default='/tmp/tensorflow/mnist/input_data', 
                    help='Directory for storing input data')

    args = pp.parse_args()
    assert args.algo in ['sgd', 'momentum', 'adam', 'rmsprop', 'hmc']

    # Set up the directory to log things.
    args.log_dir = "logs/mnist/seed_"+str(args.seed)
    print("log_dir: {}\n".format(args.log_dir))
    assert not os.path.exists(args.log_dir), "Error: log_dir already exists!"
    logz.configure_output_dir(args.log_dir)
    os.makedirs(args.log_dir+'/snapshots/') # NN weights
    with open(args.log_dir+'/args.pkl','wb') as f:
        pickle.dump(args, f)

    # Session, random seeds.
    sess = U.get_tf_session()
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Load datasets and get started.
    # data = utils.load_dataset('mnist') # Eh built-in is fine...
    data = input_data.read_data_sets(args.data_dir, one_hot=True)
    net = Net(sess, data, args)
    net.train()
    net.test()
