"""
Note: if using the MNIST dataset from TensorFlow, see this:
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/learn/python/learn/datasets/mnist.py
for the class definition.
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
np.set_printoptions(suppress=True, linewidth=180)


class Net:
    """ Builds the newtork. """

    def __init__(self, sess, data, args, data2=None):
        self.sess = sess
        self.data = data
        self.args = args
        self.data2 = data2

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

        # Handle the weights
        self.weights       = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.weights_v     = U.vars_to_vector(self.weights)
        self.num_weights   = U.get_num_weights(self.weights)
        self.new_weights_v = tf.placeholder(tf.float32, shape=[self.num_weights])
        self.update_wts_op = U.set_weights_from_vector(self.weights, self.new_weights_v)
        
        self.correct_preds = tf.equal(tf.argmax(self.y_targ_BC, 1), 
                                      tf.argmax(self.y_pred_BC, 1))
        self.accuracy = U.mean(tf.cast(self.correct_preds, tf.float32))

        # Construct objective and updaters
        self.loss = U.mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.y_targ_BC, logits=self.y_pred_BC)
        )

        # Somewhat different, to make it easier to use HMC later.
        self.updaters = []
        self.h_updaters = []
        self.grads = []

        # Turn logits into log probabilities, needed for HMC gradient.
        if args.algo == 'hmc':
            self.nb = tf.shape(self.x_BO)[0]
            self.y_targ_B     = tf.argmax(self.y_targ_BC, axis=1)
            self.log_prob_BC  = tf.nn.log_softmax(self.y_pred_BC)
            self.log_prob_B   = U.fancy_slice_2d(self.log_prob_BC, tf.range(self.nb), self.y_targ_B)
            self.log_prob_lik = U.mean(self.log_prob_B)
            self.grad_log_lik = tf.gradients(self.log_prob_lik, self.weights)

        for w in self.weights:
            grad = tf.gradients(self.loss, w)[0] # Extract the only list item.
            self.grads.append(grad)

            if args.algo == 'hmc':
                # For HMC, we need hyperparameters and their updaters.
                hp = HyperParams(alpha=args.gamma_alpha, beta=args.gamma_beta)
                self.updaters.append( HMCUpdater(w, grad, args, hp, self.sess) )
                self.h_updaters.append( 
                        HyperUpdater(w, grad, args, hp, self.sess, self.num_train) 
                )
            else:
                self.updaters.append( SGDUpdater(w, grad, args) )
                # For now this is only for SGD-based algorithms, not HMC.
                self.train_op = tf.group(*[up.update() for up in self.updaters])

        # View a summary and initialize.
        self._print_summary()
        self.sess.run(tf.global_variables_initializer())


    def hmc_update(self, xs, ys, hparams):
        """ Performs HMC update, temporary then I'll put it in classes. 
       
        Most implementations online assume we call tf.gradients() on the
        log_joint_prob for all the weights. This is the same as log p(theta,x),
        but we don't need the prior here since I already have it in closed form.
        Hence, the gradient is only for the log likelihood, which furthermore
        depends on the appropriate class.

        One key point is that we iterate through leapfrog iterations, THEN
        iterate through weights. Don't do it the reverse, because one weight
        change can affect the subsequent gradients, etc.
        """
        L = self.args.num_leapfrog
        eps = self.args.leapfrog_step
        feed = {self.x_BO: xs, self.y_targ_BC: ys}

        # TODO: put all this in a TensorFlow graph?
        # For now q_old, p_old, q_new, p_new, weights, grads are synchronized lists.
        weights = self.sess.run(self.weights)  
        q_old = np.copy(weights) # Position
        p_old = []               # Momentum
        for val in q_old:
            p_old.append( np.random.normal(size=val.shape) )
        q_new = copy.deepcopy(q_old)
        p_new = copy.deepcopy(p_old)

        grads_wrt_lik = self.sess.run(self.grad_log_lik, feed)

        for ll in range(L):
            # Half momentum and full position (later, do another half momentum).
            norms = []

            for (idx, grad) in enumerate(grads_wrt_lik):
                norms.append(np.linalg.norm(grad))
                # I think we want negative log probs then add the regularizer.
                grad = -grad + (hparams[idx][1] * q_new[idx])
                p_new[idx] = p_new[idx] - (0.5*eps) * grad
                q_new[idx] = q_new[idx] + eps * p_new[idx]

            # For the second half-momentum, we need to update our gradients. For
            # this we need to assign the weights for the old network.
            w_vec = np.concatenate([np.reshape(w,[-1]) for w in q_new], axis=0)
            self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})

            grads_wrt_lik = self.sess.run(self.grad_log_lik, feed)
            for (idx, grad) in enumerate(grads_wrt_lik):
                grad = -grad + (hparams[idx][1] * q_new[idx]) # same computation...
                p_new[idx] = p_new[idx] - (0.5*eps) * grad

        # Negating momentum is not necessary

        # M(H) Test. Full over the data. TODO: put in a class.
        H_old = 0.5 * np.sum([np.linalg.norm(w)**2 for w in p_old]) # momentum!
        H_new = 0.5 * np.sum([np.linalg.norm(w)**2 for w in p_new]) # momentum!

        for idx,(wold,wnew) in enumerate(zip(q_old, q_new)):
            H_old += -(0.5 * hparams[idx][1]) * np.linalg.norm(wold) **2
            H_new += -(0.5 * hparams[idx][1]) * np.linalg.norm(wnew) **2
        
        iters_per_epoch_train = int(self.num_train / self.args.bsize)

        # put in old net weights then get log lik for data
        w_vec = np.concatenate([np.reshape(w,[-1]) for w in q_old], axis=0)
        self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})
        for ii in range(iters_per_epoch_train):
            xs, ys = self.data2.train.next_batch(self.args.bsize)
            feed = {self.x_BO: xs, self.y_targ_BC: ys}
            H_old += self.sess.run(self.log_prob_lik, feed)

        w_vec = np.concatenate([np.reshape(w,[-1]) for w in q_new], axis=0)
        self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})
        for ii in range(iters_per_epoch_train):
            xs, ys = self.data2.train.next_batch(self.args.bsize)
            feed = {self.x_BO: xs, self.y_targ_BC: ys}
            H_new += self.sess.run(self.log_prob_lik, feed)

        print(H_new,H_old)
        test_stat = -H_new + H_old

        if (np.log(np.random.random()) < test_stat):
            # accept
            pass # do nothing, q_new is already updated
        else:
            w_vec = np.concatenate([np.reshape(w,[-1]) for w in q_old], axis=0)
            self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})

    
    def train(self):
        args = self.args
        mnist = self.data
        iters_per_epoch_train = int(self.num_train / args.bsize)
        iters_per_epoch_valid = int(self.num_valid / args.bsize)
        print("Training, num iters per epoch for train & valid: {}, {}".format(
                iters_per_epoch_train, iters_per_epoch_valid))
        t_start = time.time()

        for ee in range(args.epochs):
            # Resample the hyperparameters if we're doing HMC.
            if args.algo == 'hmc':
                hparams = []
                for hp in self.h_updaters:
                    hparams.append( hp.update() )

            for ii in range(iters_per_epoch_train):
                xs, ys = mnist.train.next_batch(args.bsize)
                feed = {self.x_BO: xs, self.y_targ_BC: ys}

                # For now just do the entire update here. Transfer to classes later.
                if args.algo == 'hmc':
                    assert len(hparams) == len(self.updaters)
                    self.hmc_update(xs, ys, hparams)
                else:
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

            # Log after each epoch, if desired.
            if (ee % args.log_every_t_epochs == 0):
                print("\n  ************ Epoch %i ************" % ee)
                elapsed_time_hours = (time.time() - t_start) / (60.0 ** 2)
                if args.algo == 'hmc':
                    for ww, hp in zip(self.weights, hparams):
                        print("{:10} -- (plambda={:.3f}, wd={:.5f})".format(
                            str(ww.get_shape().as_list()),hp[0],hp[1]))
                logz.log_tabular("ValidAcc",  acc_valid)
                logz.log_tabular("ValidLoss", loss_valid)
                logz.log_tabular("TimeHours", elapsed_time_hours)
                logz.log_tabular("Epochs",    ee)
                logz.dump_tabular()


    def test(self):
        args = self.args
        mnist = self.data
        total_iters = int(self.num_test / args.bsize)
        feed = {self.x_BO: mnist.test.images, self.y_targ_BC: mnist.test.labels}
        accuracy = self.sess.run(self.accuracy, feed)
        print("test accuracy: {}".format(accuracy))


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

        print("gradients:")
        for g in self.grads:
            shp = g.get_shape().as_list()
            print("- {} shape:{} size:{}".format(g.name, shp, np.prod(shp)))

        print("hyperparams:")
        if self.args.algo == 'hmc':
            for hu in self.h_updaters:
                print("- hp with size:{}".format(hu.size))

        print("\n=== DONE WITH SUMMARY ===\n")
