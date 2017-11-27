"""
Note: if using the MNIST dataset from TensorFlow, see this:
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/learn/python/learn/datasets/mnist.py
for the class definition.
"""
import argparse
import logz
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import utils as U
from nnupdaters import (SGDUpdater, HMCUpdater)
np.set_printoptions(suppress=True, linewidth=180)


class Net:
    """ Builds the newtork. """

    def __init__(self, sess, data, args):
        self.sess = sess
        self.data = data
        self.args = args

        # Obviously assumes we're using MNIST.
        self.x_dim = 28*28 
        self.num_classes = 10
        self.num_train = data.train.num_examples
        self.num_valid = data.validation.num_examples
        self.num_test = data.test.num_examples

        # Placeholders for input data and (known) targets, in one-hot form.
        self.x_BO      = tf.placeholder(shape=[None,self.x_dim], dtype=tf.float32)
        self.y_targ_BC = tf.placeholder(shape=[None,self.num_classes], dtype=tf.int32)

        # Build network for predictions.
        with tf.variable_scope('Classifier'):
            self.y_Bh1 = tf.nn.sigmoid(tf.layers.dense(self.x_BO, 100))
            self.y_pred_BC = tf.layers.dense(self.y_Bh1, self.num_classes)

        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.num_weights = U.get_num_weights(self.weights)
        self.correct_preds = tf.equal(tf.argmax(self.y_targ_BC, 1), 
                                      tf.argmax(self.y_pred_BC, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))

        # Construct objective and updaters
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_targ_BC, logits=self.y_pred_BC)
        )

        # Somewhat different, to make it easier to use HMC later.
        self.updaters = []
        self.h_updaters = []
        self.grads = []

        for w in self.weights:
            grad = tf.gradients(self.loss, w)[0] # Extract the only list item.
            self.grads.append(grad)

            if self.algo == 'hmc':
                # For HMC, we also need hyperparameter updates.
                # self.updaters.append( HMCUpdater(w, grad, args) )
                # self.h_updaters.append( HyperUpdater(...) )
            else:
                self.updaters.append( SGDUpdater(w, grad, args) )

        self.train_op = tf.group(*[up.update() for up in self.updaters])

        # View a summary and initialize.
        self._print_summary()
        self.sess.run(tf.global_variables_initializer())


    def train(self):
        args = self.args
        mnist = self.data
        iters_per_epoch_train = int(self.num_train / args.bsize)
        iters_per_epoch_valid = int(self.num_valid / args.bsize)
        print("Training, num iters per epoch for train & valid: {}, {}".format(
                iters_per_epoch_train, iters_per_epoch_valid))

        for ee in range(args.epochs):
            for ii in range(iters_per_epoch_train):
                xs, ys = mnist.train.next_batch(args.bsize)
                feed = {self.x_BO: xs, self.y_targ_BC: ys}
                _, grads, loss = self.sess.run([self.train_op, self.grads, self.loss], feed)

            # Check validation set performance after each epoch.
            loss_valid = 0.
            acc_valid = 0.

            for ii in range(iters_per_epoch_valid):
                xs, ys = mnist.validation.next_batch(args.bsize)
                feed = {self.x_BO: xs, self.y_targ_BC: ys}
                acc, loss = self.sess.run([self.accuracy, self.loss], feed)
                acc_valid += acc
                loss_valid += loss

            acc_valid /= iters_per_epoch_valid
            loss_valid /= iters_per_epoch_valid
            print("epoch {} done, valid loss and acc: {:.4f}, {:.4f}".format(
                ee, loss_valid, acc_valid))


    def test(self):
        args = self.args
        mnist = self.data
        total_iters = int(self.num_test / args.bsize)
        feed = {self.x_BO: mnist.test.images, self.y_targ_BC: mnist.test.labels}
        accuracy = self.sess.run(self.accuracy, feed)
        print("test accuracy: {}".format(accuracy))


    def _print_summary(self):
        print("\n=== START OF SUMMARY ===\n")
        print("Total number of weights: {}.".format(self.num_weights))

        print("weights:")
        for v in self.weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))

        print("gradients:")
        for g in self.grads:
            shp = g.get_shape().as_list()
            print("- {} shape:{} size:{}".format(g.name, shp, np.prod(shp)))

        print("\n=== DONE WITH SUMMARY ===\n")
