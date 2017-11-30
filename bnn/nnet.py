"""
Note: if using the MNIST dataset from TensorFlow, see this:
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/learn/python/learn/datasets/mnist.py
for the class definition. However, it's probably easiest just to use stuff.
"""
import argparse
import copy
import logz
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time
import utils as U
from nnupdaters import (SGDUpdater, HMCUpdater, HyperUpdater, HyperParams)
from collections import defaultdict
np.set_printoptions(suppress=True, linewidth=180)


class Net:

    def __init__(self, sess, data, args):
        self.sess = sess
        self.data = data
        self.args = args
        self.data_mb_list = U.list_of_minibatches(data, args.bsize)
        self.num_train_mbs = len(self.data_mb_list['X_train'])
        self.num_valid_mbs = len(self.data_mb_list['X_valid'])

        # Obviously assumes we're using MNIST.
        self.x_dim = 28*28 
        self.num_classes = 10
        self.num_train = self.args.data_stats['num_train']
        self.num_valid = self.args.data_stats['num_valid']
        self.num_test  = self.args.data_stats['num_test']

        # Placeholders for input data and (known) targets.
        self.x_BO      = tf.placeholder(shape=[None,self.x_dim], dtype=tf.float32)
        self.y_targ_B  = tf.placeholder(shape=[None], dtype=tf.int32)

        # Build network for predictions.
        with tf.variable_scope('Classifier'):
            self.y_Bh1 = tf.nn.sigmoid(tf.layers.dense(self.x_BO, 100))
            self.y_pred_BC = tf.layers.dense(self.y_Bh1, self.num_classes)

        self.y_pred_B      = tf.cast(tf.argmax(self.y_pred_BC, 1), tf.int32)
        self.correct_preds = tf.equal(self.y_targ_B, self.y_pred_B)
        self.accuracy      = U.mean(tf.cast(self.correct_preds, tf.float32))
        self.y_targ_BC     = tf.one_hot(indices=self.y_targ_B, depth=self.num_classes)

        # Handle the weights plus an assignment operator (useful for leapfrogs).
        self.weights       = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.num_weights   = U.get_num_weights(self.weights)
        self.weights_v     = U.vars_to_vector(self.weights)
        self.new_weights_v = tf.placeholder(tf.float32, shape=[self.num_weights])
        self.update_wts_op = U.set_weights_from_vector(self.weights, self.new_weights_v)

        # Construct objective (using our one-hot encoding) and updaters.
        self.loss = U.mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_targ_BC, logits=self.y_pred_BC)
        )

        # Somewhat different, to make it easier to use HMC later.
        # For HMC, we need hyperparameters and their updaters.
        if args.algo == 'hmc':
            self.hmc_updater = HMCUpdater(self.sess,
                                          self.args,
                                          self.x_BO,
                                          self.y_targ_B,
                                          self.y_pred_BC,
                                          self.weights,
                                          self.new_weights_v,
                                          self.update_wts_op,
                                          self.loss,
                                          self.num_train,
                                          self.data_mb_list)
        else:
            self.grads = []
            for w in self.weights:
                grad = tf.gradients(self.loss, w)[0] # Extract the only list item.
                self.grads.append(grad)
                self.updaters.append( SGDUpdater(w, grad, args) )
                self.train_op = tf.group(*[up.update() for up in self.updaters])

        # View a summary and initialize.
        self._print_summary()
        self.sess.run(tf.global_variables_initializer())

   
    def train(self):
        args = self.args
        mnist = self.data
        t_start = time.time()

        for ee in range(args.epochs):
            # Resample the hyperparameters if we're doing HMC.
            if args.algo == 'hmc':
                hparams = self.hmc_updater.update_hparams()
                hmc_info = defaultdict(list)

            for ii in range(self.num_train_mbs):
                xs = self.data_mb_list['X_train'][ii]
                ys = self.data_mb_list['y_train'][ii]
                if args.algo == 'hmc':
                    info = self.hmc_updater.hmc(xs, ys, hparams)
                    for key in info:
                        hmc_info[key].append(info[key])
                else:
                    feed = {self.x_BO: xs, self.y_targ_B: ys}
                    _, grads, loss = self.sess.run([self.train_op, self.grads, self.loss], feed)
        
            # Log after each epoch, if desired and test on validation.
            if (ee % args.log_every_t_epochs == 0):
                acc_valid, loss_valid = self._check_validation()                

                print("\n  ************ Epoch %i ************" % (ee+1))
                elapsed_time_hours = (time.time() - t_start) / (60.0 ** 2)

                if args.algo == 'hmc':
                    for ww, hp in zip(self.weights, hparams):
                        print("{:10} -- plambda={:.3f}".format(
                            str(ww.get_shape().as_list()), hp))
                    logz.log_tabular("HMCAcceptRateEpoch", np.mean(hmc_info['accept']))
                    logz.log_tabular("KineticOldMean",     np.mean(hmc_info['K_old']))
                    logz.log_tabular("KineticNewMean",     np.mean(hmc_info['K_new']))
                    logz.log_tabular("PotentialOldMean",   np.mean(hmc_info['U_old']))
                    logz.log_tabular("PotentialNewMean",   np.mean(hmc_info['U_new']))
                    logz.log_tabular("HamiltonianOldMean", np.mean(hmc_info['H_old']))
                    logz.log_tabular("HamiltonianNewMean", np.mean(hmc_info['H_new']))

                logz.log_tabular("ValidAcc",    acc_valid)
                logz.log_tabular("ValidLoss",   loss_valid)
                logz.log_tabular("Temperature", args.temperature)
                logz.log_tabular("TimeHours",   elapsed_time_hours)
                logz.log_tabular("Epochs",      ee)
                logz.dump_tabular()


    def _check_validation(self):
        """ Check validation set performance, normally after each epoch. """
        loss_valid = 0.
        acc_valid = 0.

        for ii in range(self.num_valid_mbs):
            xs = self.data_mb_list['X_valid'][ii]
            ys = self.data_mb_list['y_valid'][ii]
            feed = {self.x_BO: xs, self.y_targ_B: ys}
            acc, loss = self.sess.run([self.accuracy, self.loss], feed)
            acc_valid += acc
            loss_valid += loss

        acc_valid /= self.num_valid_mbs
        loss_valid /= self.num_valid_mbs
        return acc_valid, loss_valid


    def test(self):
        args = self.args
        mnist = self.data
        feed = {self.x_BO: self.data['X_test'], self.y_targ_B: self.data['y_test']}
        accuracy = self.sess.run(self.accuracy, feed)
        print("test set accuracy: {}".format(accuracy))


    # ---------
    # Debugging
    # ---------

    def _print_summary(self):
        print("\n=== START OF SUMMARY ===\n")
        print("Total number of weights: {}.".format(self.num_weights))

        print("weights:")
        for v in self.weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))

        if self.args.algo != 'hmc':
            print("gradients:")
            for g in self.grads:
                shp = g.get_shape().as_list()
                print("- {} shape:{} size:{}".format(g.name, shp, np.prod(shp)))

        if self.args.algo == 'hmc':
            #print("hyperparams:")
            #for hu in self.h_updaters:
            #    print("- hp with size:{}".format(hu.size))
            pass

        print("\nnum_train_mbs: {}".format(self.num_train_mbs))
        print("num_valid_mbs: {}".format(self.num_valid_mbs))

        print("\n=== DONE WITH SUMMARY ===\n")
