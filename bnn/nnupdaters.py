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
import collections
HyperParams = collections.namedtuple('HyperParams', 'alpha beta')


class SGDUpdater:
    """
    Do things slightly different from normal TF to make it more similar to the
    HMC updates. We create one tf.train.Optimizer for _each_ set of weights in
    our network. Then we apply the pre-computed gradient. All of this should be
    in the same computational graph so there shouldn't be issues.
    """

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


class HMCUpdater:
    # TODO in progress

    def __init__(self, sess, args, hparams, weights, new_weights, update_op):
        self.sess = sess
        self.args = args
        self.hparams = hparams
        self.weights = weights
        self.new_weights = new_weights
        self.update_op = update_op
        assert len(self.hparams) == len(self.weights)


    def update(self):
        """ Perform leapfrog steps here. """
        pass


class HyperUpdater:
    """
    By assumption, the hyper-parameters have IID Gamma priors, so we apply this
    to one set of weights. If we have N sets of weights total, then we have N
    instances of this class, one per weight.  Also, the prior is assumed fixed,
    so we will never change the internal alpha and gamma priors across different
    epochs.  (I thought we did once, but then it's not a prior!)
    """

    def __init__(self, w, args, hp, sess, bsize):
        self.w = w
        self.args = args
        self.hp = hp
        self.sess = sess
        self.size = np.prod( (self.w).get_shape().as_list() )
        self.bsize = bsize

    def update(self):
        """ Perform the Gibbs steps and returns weight decay. 
        
        We sample the precision term, but we only need the resulting weight
        decay as that tells us the term to add for the momentum update. I return
        plambda anyway, for debugging purposes.
        """
        weights = self.sess.run(self.w)
        sq_norm = (np.linalg.norm(weights) ** 2)
        alpha   = self.hp.alpha + 0.5 * self.size
        beta    = self.hp.beta + 0.5 * sq_norm

        plambda = np.random.gamma( alpha, 1.0 / beta )
        weight_decay = plambda / self.bsize # I _think_ self.bsize is OK ...
        return (plambda, weight_decay)
