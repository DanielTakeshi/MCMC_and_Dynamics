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
import copy
import numpy as np
import sys
import tensorflow as tf
import utils as U
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

    def __init__(self, sess, args, x_BO, y_targ_B, y_pred_BC, weights,
            new_weights_v, update_wts_op, loss, num_train, data_mb_list):
        """ The updater for Hamiltonian Monte Carlo.

        Formulas, where `pos` are the (neural network) model parameters:

        > H(pos,mom) = U(pos) + K(mom)

        > U(pos) = - log P(pos,data)
                 = - log P(pos) - log P(x1,...,xn|pos)
                 = - (log P(pos1) + ... + log P(posk)) - \sum_i log P(xi|pos)

            with log P(posi) = -0.5 * plambdai * ||posi||_2^2

        > K(mom) = 0.5 * ||mom||_2^2

        Normal Metropolis test: accept if u < exp(H_old-H_new), otherwise
        reject. Thus, with equality (or if H_old is larger) we always accept.
        Reject more often if H_new > H_old.

        Note: be careful about where I put in the temperature. It should be
        U(theta)/T = U(pos)/T but I sometimes compute the components separately
        in later code.
        """
        self.sess = sess
        self.args = args
        self.weights = weights
        self.update_wts_op = update_wts_op
        self.loss = loss
        self.y_pred_BC = y_pred_BC
        self.num_train = num_train
        self.data_mb_list = data_mb_list
        self.num_train_mbs = len(self.data_mb_list['X_train'])
        T = args.temperature

        # Placeholders
        self.x_BO = x_BO
        self.y_targ_B = y_targ_B
        self.new_weights_v = new_weights_v
        self.hparams = tf.placeholder(shape=[None], name='hparams', dtype=tf.float32)

        # Hyperparameters for HMC
        self.h_updaters = []
        for w in self.weights:
            hp = HyperParams(alpha=args.gamma_alpha, beta=args.gamma_beta)
            self.h_updaters.append( 
                    HyperUpdater(w, args, hp, self.sess, self.num_train) 
            )

        # HMC: define U(pos) as that's what we use for gradients.
        prior_l = []   # Obviously depends on our Gaussian assumptions.
        for idx,w in enumerate(self.weights):
            prior_l.append( self.hparams[idx] * U.sum(tf.square(w)) )
        self.neg_logprior = (0.5 * U.sum(prior_l)) / T

        self.nb = tf.shape(self.x_BO)[0]
        self.logprob_BC  = tf.nn.log_softmax(self.y_pred_BC)
        self.logprob_B   = U.fancy_slice_2d(self.logprob_BC, tf.range(self.nb), self.y_targ_B)
        self.logprob_B   /= T # temperature here
        self.logprob_sum = U.sum(self.logprob_B)

        # *Negation* as we want the negative log probs. Multiply by N later.
        self.neg_logprob = -U.mean(self.logprob_B)

        self.U = self.neg_logprior + (self.num_train * self.neg_logprob)
        self.U_grads = tf.gradients(self.U, self.weights)

    
    def update_hparams(self):
        """ Updates hyperparameters. """
        return [hp.update() for hp in self.h_updaters]


    def _assign(self, new_weights):
        """ Take `new_weights` (a list of np.arrays) and assign to network. """
        w_vec = np.concatenate([np.reshape(w,[-1]) for w in new_weights], axis=0)
        self.sess.run(self.update_wts_op, {self.new_weights_v: w_vec})


    def hmc(self, xs, ys, hparams):
        """ Perform one iteration of HMC. 

        Iterate through leapfrog iterations, THEN through weights. Don't do it
        in reverse, because one weight change affects the subsequent gradients.

        Parameters
        ----------
        xs, ys: [np.array, np.array]
            A minibatch of data and labels, respectively.
        
        Returns
        -------
        A dictionary containing statistics about this HMC iteration.
        """
        L = self.args.num_leapfrog
        eps = self.args.leapfrog_step
        hparams = np.array(hparams)
        feed_l = {self.x_BO: xs, self.y_targ_B: ys, self.hparams: hparams}

        # For now q_old, p_old, q_new, p_new, weights, grads are synchronized lists.
        weights = self.sess.run(self.weights)  
        pos_old = [] # Old Position
        mom_old = [] # Old Momentum
        for w in weights:
            pos_old.append( np.copy(w) )
            mom_old.append( np.random.normal(size=w.shape) )
        pos_new = copy.deepcopy(pos_old)
        mom_new = copy.deepcopy(mom_old)
        assert len(weights) == len(pos_old) == len(mom_old) == len(pos_new) \
                == len(mom_new) == len(hparams)

        # Run leapfrog. Half momentum, full position, half momentum.
        grads_wrt_U = self.sess.run(self.U_grads, feed_l)

        for ll in range(L):
            for (idx, grad) in enumerate(grads_wrt_U):
                mom_new[idx] += -(0.5*eps) * grad
                pos_new[idx] += eps * mom_new[idx]

            # Get new gradients.
            self._assign(new_weights=pos_new)
            grads_wrt_U = self.sess.run(self.U_grads, feed_l)

            for (idx, grad) in enumerate(grads_wrt_U):
                mom_new[idx] += -(0.5*eps) * grad

        # Negate momentum. Not actually necessary w/our kinetic energy, I think.
        mom_new = [-m for m in mom_new]

        # ----------------------------------------------------------------------
        # M(H) Test. Full over the data. First, handle momentum.
        # Later: put this in my custom class.
        # ----------------------------------------------------------------------
        K_old = 0.5 * np.sum([np.linalg.norm(w)**2 for w in mom_old])
        K_new = 0.5 * np.sum([np.linalg.norm(w)**2 for w in mom_new])

        # Handle U(theta) now. First, handle CURRENT/NEW weights.
        U_new = self.sess.run(self.neg_logprior, {self.hparams: hparams})
        for ii in range(self.num_train_mbs):
            xs = self.data_mb_list['X_train'][ii]
            ys = self.data_mb_list['y_train'][ii]
            feed = {self.x_BO: xs, self.y_targ_B: ys, self.hparams: hparams}
            U_new -= self.sess.run(self.logprob_sum,feed)

        # Now do U(theta_old). Assign theta_old weights.
        self._assign(new_weights=pos_old)
        U_old = self.sess.run(self.neg_logprior, {self.hparams: hparams})

        for ii in range(self.num_train_mbs):
            xs = self.data_mb_list['X_train'][ii]
            ys = self.data_mb_list['y_train'][ii]
            feed = {self.x_BO: xs, self.y_targ_B: ys, self.hparams: hparams}
            U_old -= self.sess.run(self.logprob_sum, feed)

        # Collect information.
        H_old = U_old + K_old
        H_new = U_new + K_new
        test_stat = -H_new + H_old

        if (np.log(np.random.random()) < test_stat):
            self._assign(new_weights=pos_new)
            accept = 1
        else:
            accept = 0
        print(H_old,H_new,accept)

        info = {'accept': accept, 'K_old':K_old, 'K_new':K_new, 'U_old':U_old,
                'U_new':U_new, 'H_old':H_old, 'H_new':H_new}
        return info


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
        plambda anyway, for debugging purposes. Update: not anymore ...

        Update: I'm ignoring the self.bsize ... because I did the math and it
        doesn't seem to show up anywhere ...
        """
        weights = self.sess.run(self.w)
        sq_norm = (np.linalg.norm(weights) ** 2)
        alpha   = self.hp.alpha + 0.5 * self.size
        beta    = self.hp.beta + 0.5 * sq_norm

        plambda = np.random.gamma( alpha, 1.0 / beta )
        # weight_decay = plambda / self.bsize # I _think_ self.bsize is OK ...
        weight_decay = plambda
        return weight_decay
