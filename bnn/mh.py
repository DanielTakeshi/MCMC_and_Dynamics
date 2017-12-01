"""
Metropolis-Hastings tests. (Well, technically Metropolis-only.)
"""

import numpy as np
import sys
import utils as U


class MHTest:

    def __init__(self, params):
        # Bells and whistles
        self.sess = params['sess']
        self.args = params['args']
        self.weights = params['weights']
        self.data_mb_list = params['data_mb_list']
        self.num_train_mbs = len(self.data_mb_list['X_train'])
        self.update_wts_op = params['update_wts_op']

        # Placeholders
        self.x_BO = params['x_BO']
        self.y_targ_B = params['y_targ_B']
        self.new_weights_v = params['new_weights_v']
        self.hparams = params['hparams']

        # Stuff computed from HMC class.
        self.neg_logprior = params['neg_logprior']
        self.logprob_sum = params['logprob_sum']


    def test(self, hparams, mom_old, mom_new, pos_old, pos_new):
        """ Runs the Metropolis test. Subclasses must override this.

        Parameters
        ----------
        hparams: [np.array]
            Numpy array containing the \lambda hyperparameters, i.e. the
            Gaussian precision terms for the neural network components.
        mom_old, mom_new: [list, list]
            Lists that contain numpy weight arrays for each of the momentum
            auxiliary variables in HMC.
        pos_old, pos_new: [list, list]
            Lists that contain numpy weight arrays for each of the weights in
            the network.

        Returns
        -------
        Dictionary containing: acceptance result along with six values: H_old,
        H_new, U_old, U_new, K_old, K_new where H is the Hamiltonian, which is
        equivalent to H = U + K, and U and K are the potential and kinetic
        energies, respectively. These are computed for both the old "current"
        sample and the new "proposed" sample.
        """
        raise NotImplementedError


    def _kinetic_energy(self, mom_old, mom_new):
        """ Computes K(mom_old) and K(mom_new) assuming quadratic potential with
        an identity mass matrix.

        Parameters
        ----------
        mom_old, mom_new: [list, list]
            Lists that contain numpy weight arrays for each of the momentum
            auxiliary variables in HMC.
        """
        K_old = 0.5 * np.sum([np.linalg.norm(w)**2 for w in mom_old])
        K_new = 0.5 * np.sum([np.linalg.norm(w)**2 for w in mom_new])
        return K_old, K_new


class MHNormal(MHTest):
    """
    The normal Metropolis test which relies on the full data log likelihood.
    It's mainly for a costly baseline and as a sanity check.
    """

    def __init__(self, params):
        super().__init__(params)


    def test(self, hparams, mom_old, mom_new, pos_old, pos_new):
        """
        As this is the normal MH test, we must iterate through the entire
        dataset when computing the potential energy U(theta_old) and
        U(theta_new).
        """
        K_old, K_new = self._kinetic_energy(mom_old, mom_new)

        U_new = self.sess.run(self.neg_logprior, {self.hparams: hparams})
        for ii in range(self.num_train_mbs):
            xs = self.data_mb_list['X_train'][ii]
            ys = self.data_mb_list['y_train'][ii]
            feed = {self.x_BO: xs, self.y_targ_B: ys, self.hparams: hparams}
            U_new -= self.sess.run(self.logprob_sum,feed)

        # Now do U(theta_old). Assign theta_old weights.
        U.assign(self.sess, self.update_wts_op, self.new_weights_v, pos_old)

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
            U.assign(self.sess, self.update_wts_op, self.new_weights_v, pos_new)
            accept = 1
        else:
            accept = 0
        print(H_old,H_new,accept)

        info = {'accept': accept, 'K_old':K_old, 'K_new':K_new, 'U_old':U_old,
                'U_new':U_new, 'H_old':H_old, 'H_new':H_new}
        return info


class MHMinibatch(MHTest):
    """ The method proposed in:

        Seita, Pan, Chen, Canny. "An Efficient Minibatch Acceptance Test for
        Metropolis-Hastings". UAI 2017.
    """

    # TODO: implement this clas.

    def __init__(self):
        super().__init__()


    def test(self, hparams, mom_old, mom_new, pos_old, pos_new):
        K_old, K_new = self._kinetic_energy(mom_old, mom_new)


class MHSubLhd(MHTest):
    """ The method proposed in:
    
        Bardenet, Doucet, and Holmes. "Towards Scaling up Markov Chain Monte
        Carlo: An Adaptive Subsampling Approach.", ICML 2014.

    I assume we will have to explicitly compute the error bounds since we can't
    do these analytically with neural networks.
    """

    # TODO: implement this class.

    def __init__(self):
        super().__init__()


    def test(self, hparams, mom_old, mom_new, pos_old, pos_new):
        K_old, K_new = self._kinetic_energy(mom_old, mom_new)


class AustereMH(MHTest):
    """ The method proposed in:

        Korattikara, Chen, Welling. "Austerity in MCMC Land: Cutting the
        Metropolis-Hastings Budget." ICML 2014.

    Note: there are two versions as described in (Seita et al., 2017),
    conservative (c) and non-conservative (nc). 
    """

    # TODO: implement this class.

    def __init__(self, conservative):
        super().__init__()
        self.conservative = conservative


    def test(self, hparams, mom_old, mom_new, pos_old, pos_new):
        K_old, K_new = self._kinetic_energy(mom_old, mom_new)
