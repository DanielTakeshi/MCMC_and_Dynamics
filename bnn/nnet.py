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
            self.hparams = tf.placeholder(shape=[None], dtype=tf.float32)
            self.h_updaters = []
            for w in self.weights:
                hp = HyperParams(alpha=args.gamma_alpha, beta=args.gamma_beta)
                self.h_updaters.append( 
                        HyperUpdater(w, args, hp, self.sess, self.num_train) 
                ) # Indeed I think `self.num_train`, not `args.bsize`.

            ## # Finally the HMC class. We'll pass in self.hparams as a
            ## # placeholder in the feed-dict.
            ## self.HMC = HMCUpdater(self.sess, args, self.hparams, self.weights,
            ##         self.new_weights_v, self.update_wts_op)

            self.nb = tf.shape(self.x_BO)[0]
            self.logprob_BC  = tf.nn.log_softmax(self.y_pred_BC)
            self.logprob_B   = U.fancy_slice_2d(self.logprob_BC, tf.range(self.nb), self.y_targ_B)
            self.logprob_lik = U.mean(self.logprob_B)
            self.grad_loglik = tf.gradients(self.logprob_lik, self.weights)

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


    def hmc_update(self, xs, ys, hparams):
        """ Performs HMC update, temporary then I'll put it in classes. 

        This is one "update" so it must take one full step (or reject and reuse
        the old one).
       
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
        feed = {self.x_BO: xs, self.y_targ_B: ys}

        # For now q_old, p_old, q_new, p_new, weights, grads are synchronized lists.
        weights = self.sess.run(self.weights)  
        pos_old = [] # Old Position
        mom_old = [] # Old Momentum
        for w in weights:
            pos_old.append( np.copy(w) )
            mom_old.append( np.random.normal(size=w.shape) )
        pos_new = copy.deepcopy(pos_old)
        mom_new = copy.deepcopy(mom_old)
        assert len(weights) == len(pos_old) == len(mom_old) == len(pos_new) == len(mom_new)

        # Quick sanity checks, gradients shouldn't be zero. For some of the
        # _input_ layer it's 0 if the images are all 0 in those spots.
        grads_wrt_lik = self.sess.run(self.grad_loglik, feed)
        for g in grads_wrt_lik:
            assert np.linalg.norm(g) > 1e-5
            assert np.mean(np.abs(g)) > 1e-5

        # Finally, leapfrogs. Half momentum, full position, half momentum.
        for ll in range(L):
            for (idx, grad) in enumerate(grads_wrt_lik):
                # I think we want negative log probs then add the regularizer.
                grad = -grad + (hparams[idx][1] * pos_new[idx])
                mom_new[idx] += -(0.5*eps) * grad
                pos_new[idx] += eps * mom_new[idx]

            # For the second half-momentum, we need to update our gradients. For
            # this we need to assign (i.e., update) the weights of the newtwork.
            w_vec = np.concatenate([np.reshape(w,[-1]) for w in pos_new], axis=0)
            self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})
            grads_wrt_lik = self.sess.run(self.grad_loglik, feed)

            for (idx, grad) in enumerate(grads_wrt_lik):
                # Same computation, with pos_new which was updated earlier.
                grad = -grad + (hparams[idx][1] * pos_new[idx])
                mom_new[idx] += -(0.5*eps) * grad


        # Negate the momentum at the end. Not actually necessary with our
        # kinetic energy formulation, I think ...
        for idx in range(len(mom_new)):
            mom_new[idx] = -mom_new[idx]

        # M(H) Test. Full over the data. First, handle momentum.
        K_old = 0.5 * np.sum([np.linalg.norm(w)**2 for w in mom_old])
        K_new = 0.5 * np.sum([np.linalg.norm(w)**2 for w in mom_new])

        # Handle U(theta) now. First, -log P(theta), the priors.
        U_old = 0.0
        U_new = 0.0

        for idx,(wold,wnew) in enumerate(zip(pos_old,pos_new)):
            # Pretty sure plambda is the same for both
            plambda = hparams[idx][1]   
            U_old += -(0.5*plambda) * np.linalg.norm(wold)**2
            U_new += -(0.5*plambda) * np.linalg.norm(wnew)**2

        # Let's go through the network with its CURRENT weights (which are the
        # current positions since I assigned them earlier!). This computes the
        # *negative* log prob of data given param, so `-log P(D|theta)`.
        for ii in range(self.num_train_mbs):
            xs = self.data_mb_list['X_train'][ii]
            ys = self.data_mb_list['y_train'][ii]
            feed = {self.x_BO: xs, self.y_targ_B: ys}
            U_new +=  - np.sum( self.sess.run(self.logprob_B,feed) )

        # Put in OLD neural network weights.
        w_vec = np.concatenate([np.reshape(w,[-1]) for w in pos_old], axis=0)
        self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})

        for ii in range(self.num_train_mbs):
            xs = self.data_mb_list['X_train'][ii]
            ys = self.data_mb_list['y_train'][ii]
            feed = {self.x_BO: xs, self.y_targ_B: ys}
            U_old +=  - np.sum( self.sess.run(self.logprob_B,feed) )

        H_old = K_old + U_old
        H_new = K_new + U_new
        test_stat = -H_new + H_old

        if (np.log(np.random.random()) < test_stat):
            w_vec = np.concatenate([np.reshape(w,[-1]) for w in pos_new], axis=0)
            self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})
            accept = 1
        else:
            accept = 0
        print(H_old,H_new,accept)

        info = {'accept': accept}
        return info

    
    def train(self):
        args = self.args
        mnist = self.data
        t_start = time.time()

        for ee in range(args.epochs):
            # Resample the hyperparameters if we're doing HMC.
            if args.algo == 'hmc':
                hparams = []
                for hp in self.h_updaters:
                    hparams.append( hp.update() )

            for ii in range(self.num_train_mbs):
                xs = self.data_mb_list['X_train'][ii]
                ys = self.data_mb_list['y_train'][ii]
                if args.algo == 'hmc':
                    hmc_info = self.hmc_update(xs, ys, hparams)
                else:
                    feed = {self.x_BO: xs, self.y_targ_B: ys}
                    _, grads, loss = self.sess.run([self.train_op, self.grads, self.loss], feed)

            # Check validation set performance after each epoch.
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

            # Log after each epoch, if desired.
            if (ee % args.log_every_t_epochs == 0):
                print("\n  ************ Epoch %i ************" % (ee+1))
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

        if self.args.algo != 'hmc':
            print("gradients:")
            for g in self.grads:
                shp = g.get_shape().as_list()
                print("- {} shape:{} size:{}".format(g.name, shp, np.prod(shp)))

        if self.args.algo == 'hmc':
            print("hyperparams:")
            for hu in self.h_updaters:
                print("- hp with size:{}".format(hu.size))

        print("\n=== DONE WITH SUMMARY ===\n")
