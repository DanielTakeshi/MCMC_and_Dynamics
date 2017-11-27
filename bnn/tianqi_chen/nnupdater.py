"""
  Implementation of neutral network 
  parameter update method
  Tianqi Chen
"""
import numpy as np
import sys

class SGDUpdater:
    """ updater that performs SGD update given weight parameter """

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
        self.m_w[:] *= ( 1.0 - param.mdecay ) # Ignore during SGLD.
        self.m_w[:] += (-param.eta) * ( self.g_w + self.wd * self.w )
        if param.need_sample():
            # E.g. during SGLD this is the Gaussian noise for exploration.
            self.m_w[:] += np.random.randn( self.w.size ).reshape( self.w.shape ) * param.get_sigma()
        # Weights are `self.w`, updated from the computed momentums.
        self.w[:] += self.m_w
        

class NAGUpdater:
    """ updater that performs NAG(nestrov's momentum) update given weight parameter """

    def __init__( self, w, g_w, param ):
        self.w = w
        self.g_w = g_w
        # updater specific weight decay
        self.wd = param.wd
        self.param = param
        self.m_w = np.zeros_like( w )
        self.m_old = np.zeros_like( w )

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
        Update hyper parameters. The stuff on the Gibbs step is most important. 
        """
        param = self.param
        if not param.need_hsample():
            return

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
            # note: use the shape/rate parameterization. However we need to
            # invert the rate parameter to turn it into a scale parameter.
            # This is only for ONE lambda (so one of lambda_A OR lambda_B, etc.,
            # from the SGHMC paper) since this is a scalar, obviously.
            plambda = np.random.gamma( alpha, 1.0 / beta )

        # Set new weight decay, equivalent to Gaussian prior on weights.
        wd = plambda / param.num_train
        for u in self.updaterlist:
            u.wd = wd

        ss = ','.join( str(u.w.shape) for u in self.updaterlist )
        print('hyperupdate[%s]:plambda=%f,wd=%f' % ( ss, plambda, wd ))
        sys.stdout.flush()


    def print_info( self ):
        return
