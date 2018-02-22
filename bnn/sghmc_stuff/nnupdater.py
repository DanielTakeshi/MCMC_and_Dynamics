"""
  Implementation of neutral network 
  parameter update method
  Tianqi Chen
"""
import numpy as np
import sys

class SGDUpdater:
    """ updater that performs SGD update given weight parameter 
    
    This is normal SGD if param.mdecay = 1.0, and momentum if othewise. Usually
    we set param.mdecay to be 0.01, I think. That way the **momentum** variable
    is one minus that ... and it's usually 0.99.
    """

    def __init__( self, w, g_w, param ):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )

    def print_info( self ):
        return

    def update( self ):
        param = self.param
        self.m_w[:] *= ( 1.0 - param.mdecay )
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w )
        self.w[:]   += self.m_w
        

class SGHMCUpdater:
    """ 
    Updater that performs update given weight parameter using SGHMC/SGLD.
    Only difference btwn the two is that SGLD explicitly sets param.mdecay=1. 
    Also, in practice I assume he set M = I*alpha where the alpha is absorbed
    into the epsilon learning rate. This way, after updating momentum, the theta
    update is literally theta_{t+1} <- theta_t + eps_k * r_t where r_t is
    momentum.

    Update: yes I did the math. Whew, it took a while but was worth it. :-)
    """

    def __init__( self, w, g_w, param ):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )

    def print_info( self ):
        return

    def update( self ):
        """ Performs the SGHMC (or SGLD) update.

        - By default, self.wd is set to starting value for first epoch.
        - The (1-mdecay) is ignored during SGLD, see `nncfg.py`.
        - Gaussian prior happens with wd * w.
        - need_sample: For SGLD this is the Gaussian noise for exploration.
        - need_sample: For SGHMC this is the exgtra Gaussian noise added.
        - need_sample is true for all epochs _after_ the first one, at index 0.
          Well, by default...

        Most of the effort is with updating momentum, since the actual _weight_
        update is simple given the momentums.
        """
        param = self.param
        self.m_w[:] *= ( 1.0 - param.mdecay ) 
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w ) 
        if param.need_sample():
            self.m_w[:] += np.random.randn(self.w.size).reshape(self.w.shape) * param.get_sigma()
        self.w[:] += self.m_w
        

class NAGUpdater:
    """ updater that performs NAG(nestrov's momentum) update given weight parameter 
    
    Question: seems like this is similar to SGHMC ... due to the added Gaussian
    noise? Yeah, my guess is this is supposed to make SGHMC have a Nesterov-like
    update. Gah.
    """

    def __init__( self, w, g_w, param ):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )
        self.m_old = np.zeros_like( w ) # Only difference with SGHMC

    def print_info( self ):
        return

    def update( self ):
        param = self.param
        momentum = 1.0 - param.mdecay
        self.m_old[:] = self.m_w
        self.m_w[:] *= momentum
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w )
        if param.need_sample():
            self.m_w[:] += np.random.randn( self.w.size ).reshape( self.w.shape ) * param.get_sigma()
        self.w[:]   += (1.0+momentum) * self.m_w - momentum * self.m_old


class HyperUpdater:
    """ Hyper Parameter Gibbs Gamma sampler for regularizer update. """

    def __init__(self, param, updaterlist):
        """ The updaterlist contains all the _regular_ hyperparameters. """
        self.updaterlist = updaterlist
        self.param = param
        self.scounter = 0
        

    def update( self ):
        """ 
        Update hyper parameters, which here results in the weight decay params.
        Note the Gibbs step. These are then fixed for the next epoch.
        """
        # Do not update hyperparameters during first epoch (index=0).
        param = self.param
        if not param.need_hsample():
            return

        # Only update hyperparameters every `param.gap_hcounter() = 100`
        # iterations which corespond to the _end_ of each epoch.
        self.scounter += 1
        if self.scounter % param.gap_hcounter() != 0:
            return
        else:
            self.scounter = 0
        
        # `u.w` are some neural network weights. By default, self.updaterlist
        # has only one element in it so we do this separately.
        sumsqr = sum( np.sum( u.w * u.w ) for u in self.updaterlist )
        sumcnt = sum( u.w.size for u in self.updaterlist )

        # Conjugate update for Gammas. (See Wikipedia or my blog post if confused.)
        alpha = param.hyper_alpha + 0.5 * sumcnt
        beta  = param.hyper_beta + 0.5 * sumsqr
        
        if param.temp < 1e-6:
            # if we are doing MAP, take the mode, 
            # note: normally MAP adjust is not as well as MCMC
            plambda = max( alpha - 1.0, 0.0 ) / beta
        else:
            # Use the shape/rate parameterization. However we need to invert the
            # rate parameter to turn it into a scale parameter.  This is only
            # for ONE lambda (so one of lambda_A OR lambda_B, etc., from the
            # SGHMC paper) since this is a scalar, obviously.
            plambda = np.random.gamma( alpha, 1.0 / beta )

        # Set new weight decay, equivalent to Gaussian prior on weights. DIVIDE
        # BY THE NUMBER OF TRAINING POINTS, which is 50k for MNIST by default.
        wd = plambda / param.num_train
        for u in self.updaterlist:
            u.wd = wd

        ss = ','.join( str(u.w.shape) for u in self.updaterlist )
        print('hyperupdate[%s]:plambda=%f,wd=%f' % ( ss, plambda, wd ))
        sys.stdout.flush()


    def print_info( self ):
        return
