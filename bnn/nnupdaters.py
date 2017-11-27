""" 
Updaters. In TensorFlow, we define a loss, `self.loss`, and we normally do
something like this, where `self.weights` are a set of trainable weights from a
neural network:

    self.optim = tf.train.GradientDescentOptimizer(args.lrate_sgd)
    self.grads, self.vars = zip(*(self.optim).compute_gradients(self.loss, self.weights))
    // clip gradients here if desired ...
    self.train_op = (self.optim).apply_gradients(zip(self.grads, self.vars))

(Actually, sometimes we don't clip and just call minimize directly on the
`tf.train.GradientDescentOptimizer` class.) However, the alternative is to
explicitly compute the gradient alone:

    self.grads = tf.gradients(self.loss, self.weights)

So that we can get more control. This should result in the same gradients. As
the documentation says:

    TensorFlow provides functions to compute the derivatives for a given
    TensorFlow computation graph, adding operations to the graph. The optimizer
    classes automatically compute derivatives on your graph, but creators of new
    Optimizers or expert users can call the lower-level functions below. [Such
    as `tf.gradients` ...]

Source: https://www.tensorflow.org/api_guides/python/train#Gradient_Computation
"""
import numpy as np
import sys
import tensorflow as tf


class SGDUpdater:

    def __init__(self, w, g_w, args):
        self.w = w
        self.g_w = g_w
        self.wd = args.wdecay

        if args.algo == 'sgd':
            self.optim = tf.train.GradientDescentOptimizer(args.lrate_sgd)
        elif args.algo == 'momentum':
            self.optim = tf.train.MomentumOptimizer(args.lrate_sgd, args.momentum)
        elif args.algo == 'adam':
            self.optim = tf.train.AdamOptimizer(args.lrate_sgd)
        elif args.algo == 'rmsprop':
            self.optim = tf.train.RMSPropOptimizer(args.lrate_sgd)
        else:
            raise ValueError()

        self.op = (self.optim).apply_gradients( [(g_w, w)] )

    def update(self):
        return self.op


#TODO fix below, add hyperparameter updater

class HMCUpdater:

    def __init__(self, w, g_w, cfg):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )

    def update(self):
        param = self.param
        self.m_w[:] *= ( 1.0 - param.mdecay ) # Ignore during SGLD.
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w )
        if param.need_sample():
            # E.g. during SGLD this is the Gaussian noise for exploration.
            self.m_w[:] += np.random.randn(self.w.size).reshape(self.w.shape) * param.get_sigma()
        # Weights are `self.w`, updated from the computed momentums.
        self.w[:] += self.m_w
