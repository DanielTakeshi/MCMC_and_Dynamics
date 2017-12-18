"""
  Implementation of neural network 
  Core implementations
  Tianqi Chen
"""
import numpy as np
import sys
np.set_printoptions(suppress=True, edgeitems=10, linewidth=180)


# Full connected layer
# note: all memory are pre-allocated, always use a[:]= instead of a= in assignment
class FullLayer:

    def __init__( self, i_node, o_node, init_sigma, rec_gsqr = False ):
        assert i_node.shape[0] == o_node.shape[0]
        self.rec_gsqr = rec_gsqr
        # node value
        self.i_node = i_node
        self.o_node = o_node
        # weight
        self.o2i_edge = np.float32(np.random.randn(i_node.shape[1], o_node.shape[1]) * init_sigma)
        self.o2i_bias = np.zeros(o_node.shape[1], 'float32') 
        # gradient
        self.g_o2i_edge = np.zeros_like( self.o2i_edge )
        self.g_o2i_bias = np.zeros_like( self.o2i_bias )
        # gradient square
        self.sg_o2i_edge = np.zeros_like( self.o2i_edge )
        self.sg_o2i_bias = np.zeros_like( self.o2i_bias )
        if self.rec_gsqr:
            self.i_square = np.zeros_like( self.i_node )
            self.o_square = np.zeros_like( self.o_node )

    def forward( self, istrain = True ):
        # forward prop, node value to o_node
        self.o_node[:] = np.dot( self.i_node, self.o2i_edge ) + self.o2i_bias

    def backprop( self, passgrad = True ):
        # backprop, gradient is stored in o_node
        # divide by batch size (Daniel: REMEMBER this division by batch size!)
        bscale = 1.0 / self.o_node.shape[0]
        self.g_o2i_edge[:] = bscale * np.dot( self.i_node.T, self.o_node )
        self.g_o2i_bias[:] = np.mean( self.o_node, 0 )
        
        # record second moment of gradient if needed (Daniel: never used)
        if self.rec_gsqr:
            self.o_square[:] = np.square( self.o_node )
            self.i_square[:] = np.square( self.i_node )
            self.sg_o2i_edge[:] = bscale * np.dot( self.i_square.T, self.o_square )
            self.sg_o2i_bias[:] = np.mean( self.o_square, 0 )
        
        # backprop to i_node if necessary
        if passgrad:
            self.i_node[:] = np.dot( self.o_node, self.o2i_edge.T )
            
    def params( self ):
        # return a reference list of parameters
        return [ (self.o2i_edge, self.g_o2i_edge, self.sg_o2i_edge), (self.o2i_bias,self.g_o2i_bias,self.sg_o2i_bias) ]


class ActiveLayer:

    def __init__( self, i_node, o_node, n_type = 'relu' ):
        assert i_node.shape[0] == o_node.shape[0]
        # node value
        self.n_type = n_type
        self.i_node = i_node
        self.o_node = o_node

    def forward( self, istrain = True ):
        # also get gradient ready in i node
        if self.n_type == 'relu':
            self.o_node[:] = np.maximum( self.i_node, 0.0 )
            self.i_node[:] = np.sign( self.o_node )
        elif self.n_type == 'tanh':
            self.o_node[:] = np.tanh( self.i_node )
            self.i_node[:] = ( 1.0 - np.square(self.o_node) )
        elif self.n_type == 'sigmoid':
            self.o_node[:] = 1.0 / ( 1.0 + np.exp( - self.i_node ) )
            self.i_node[:] = self.o_node * (1.0 - self.o_node)
        else:
            raise('NNConfig', 'unknown node_type')
        
    def backprop( self, passgrad = True ):
        if passgrad:
            self.i_node[:] *= self.o_node;
            
    def params( self ):
        return []


class SoftmaxLayer:

    def __init__( self, i_node, o_label ):
        assert i_node.shape[0] == o_label.shape[0]
        assert len( o_label.shape ) == 1
        self.i_node  = i_node
        self.o_label = o_label

    def forward( self, istrain = True ):        
        nbatch = self.i_node.shape[0]
        self.i_node[:] = np.exp( self.i_node - np.max( self.i_node, 1 ).reshape( nbatch, 1 ) )
        self.i_node[:] = self.i_node / np.sum( self.i_node, 1 ).reshape( nbatch, 1 )

    def backprop( self, passgrad = True ):
        if passgrad:
            nbatch = self.i_node.shape[0]
            for i in range( nbatch ):
                self.i_node[ i, self.o_label[i] ] -= 1.0 

    def params( self ):
        return []


class RegressionLayer:

    def __init__( self, i_node, o_label, param ):
        assert i_node.shape[0] == o_label.shape[0]
        assert i_node.shape[0] == o_label.size
        assert i_node.shape[1] == 1
        self.i_tmp  = np.zeros_like( i_node )
        self.n_type = param.out_type
        self.i_node  = i_node
        self.o_label = o_label
        self.param = param
        self.base_score = None

    def init_params( self ):
        if self.base_score != None:
            return
        param = self.param
        self.scale = param.max_label - param.min_label;
        self.min_label = param.min_label
        self.base_score = (param.avg_label - param.min_label) / self.scale
        if self.n_type == 'logistic':
            self.base_score = - math.log( 1.0 / self.base_score - 1.0 );
        print('range=[%f,%f], base=%f' %( self.min_label, param.max_label, param.avg_label ))

    def forward( self, istrain = True ):     
        self.init_params()
        nbatch = self.i_node.shape[0]
        self.i_node[:] += self.base_score
        if self.n_type == 'logistic':
            self.i_node[:] = 1.0 / ( 1.0 + np.exp( -self.i_node ) )
        self.i_tmp[:] = self.i_node[:]
        # transform to appropriate output
        self.i_node[:] = self.i_node * self.scale + self.min_label
        
    def backprop( self, passgrad = True ):
        if passgrad:
            nbatch = self.i_node.shape[0]
            label = (self.o_label.reshape( nbatch, 1 ) - self.min_label) / self.scale
            self.i_node[:] = self.i_tmp[:] - label
            #print(np.sum( np.sum( (label - self.i_tmp[:])**2 ) ))
            
    def params( self ):
        return []


class NNetwork:

    def __init__( self, layers, nodes, o_label, factory ):
        self.nodes   = nodes
        self.o_label = o_label
        self.i_node = nodes[0]
        self.o_node = nodes[-1]
        self.layers = layers
        self.weights = []
        self.updaters = []
        for l in layers:
            self.weights += l.params()            
        for w, g_w, sg_w in self.weights:
            assert w.shape == g_w.shape and w.shape == sg_w.shape
            self.updaters.append( factory.create_updater( w, g_w, sg_w ) )
        self.updaters = factory.create_hyperupdater( self.updaters ) + self.updaters

    def update( self, xdata, ylabel ):
        """ Update based on one minibatch of data. 
        
        After each minibatch goes forward and backwards to get gradients, we
        update the updaters (not hyperparameters).
        """
        self.i_node[:] = xdata
        for i in range( len(self.layers) ):
            self.layers[i].forward( True )
        self.o_label[:] = ylabel
        for i in reversed( range( len(self.layers) ) ):
            self.layers[i].backprop( i!= 0 )
        for u in self.updaters:
            u.update()

    def update_all( self, xdatas, ylabels ):
        """ Called by external code in `mnist.py`, starts the pipeline. """
        for i in range( xdatas.shape[0] ):            
            self.update( xdatas[i], ylabels[i] )
        for u in self.updaters:
            u.print_info()

    def predict( self, xdata ):
        self.i_node[:] = xdata
        for i in range( len(self.layers) ):
            self.layers[i].forward( False )
        return self.o_node


class NNEvaluator:
    """ 
    Evaluator to evaluate results. One instance of this class corresponds to the
    training, validation, or testing setups.  
    """

    def __init__( self, nnet, xdatas, ylabels, param, prefix='' ):
        self.nnet = nnet
        self.xdatas  = xdatas
        self.ylabels = ylabels
        self.param = param
        self.prefix = prefix
        nbatch, nclass = nnet.o_node.shape
        assert xdatas.shape[0] == ylabels.shape[0]
        assert nbatch == xdatas.shape[1]
        assert nbatch == ylabels.shape[1]
        # By default, of shape (100,500,10).
        # We have 100 minibatches, each with 500 elements and 10 classes.
        self.o_pred  = np.zeros( ( xdatas.shape[0], nbatch, nclass ), 'float32'  )
        self.rcounter = 0
        self.sum_wsample = 0.0


    def __get_alpha( self ):
        """
        Daniel: by default, ignore the first 50 samples as burn-in so for the
        first 50 epochs we simply get alpha = 1.0. (Also, I think Tianqi refers
        to a sample as one epoch ... so there are 750 samples we really track
        for the SGHMC results.) Then we multiply the output predictions by
        1-alpha which is ... 0 to start. I guess that's fine, we override later.

        After the burn-in, we multiply our predictions by our alpha parameter
        which will decrease their values. This should not change raw accuracy,
        but it will change the neg-log-lik. By default, param.wsample=1 so that
        this turns into 1/2 for epoch 50, then 1/3 for epoch 51, then 1/4 for
        epoch 52, etc.  I see, it looks like he does moving averages.

        This alpha is NOT to be confused with the alpha used for momentum decay,
        the `1-alpha = mu` term!!
        """
        if self.rcounter < self.param.num_burn:
            return 1.0
        else:
            self.sum_wsample += self.param.wsample
            return self.param.wsample / self.sum_wsample
        

    def eval( self, rcounter, fo ):
        """ Prints out evaluation of train, test, or valid after each epoch.

        The outer for loop predicts on _minibatches_. By default:

            > self.xdatas[i].shape: (nbatch,784)
            > self.o_pred[i,:].shape: (nbatch,10)

        with nbatch=500. Also, the predictions are **normalized**, it's NOT
        logits but the actual softmax-ed and normalized values. Without the `i`
        index, these would have a new dimension of size 100 by default, the
        number of minibatches total.

        Number of incorrect predictions: `sum_bad`, so use that for error.
        Also return negative log likelihood, which we want to _minimize_.
        Iterate through the number of elements in this minibatch and compute:

            > log P(y_i | x_i, theta)

        Then both of these are averaged over the number of total samples.

        After 50 burn-in epochs, we start doing a moving average by storing the
        previous predictions and getting new predictions by:

            > preds = (1-alpha)*prev_preds + alpha*new_preds

        with alpha annealed from 1/n where n is the number of epochs **after**
        the burn-in period, so n=1 at epoch 50, etc. Thus, by the time we're at
        epoch 800, new predictions are weighed so little. This methodology means
        that our predictions at epoch K are the AVERAGE across the K epochs thus
        far (again, not counting burn-in epochs).

        BTW this is the same thing as taking every 100 samples, because there
        are 100 minibatches, but we only take the one that appears at the end of
        each epoch.
        """
        self.rcounter = rcounter
        alpha = self.__get_alpha()        
        self.o_pred[:] *= ( 1.0 - alpha )
        sum_bad = 0.0
        sum_loglike = 0.0

        for i in range(self.xdatas.shape[0]):
            self.o_pred[i,:] += alpha * self.nnet.predict( self.xdatas[i] )
            y_pred = np.argmax( self.o_pred[i,:], 1 )            
            sum_bad += np.sum( y_pred != self.ylabels[i,:] )
            for j in range(self.xdatas.shape[1]):
                sum_loglike += np.log( self.o_pred[i, j, self.ylabels[i,j]] )

        ninst = self.ylabels.size
        fo.write( ' %s-err:%f %s-nlik:%f' % 
                ( self.prefix, sum_bad/ninst, self.prefix, -sum_loglike/ninst) )


class NNParam:
    """ Called during main `mnist.py` so that we get a set of hyperparams. """

    def __init__( self ):
        """ 
        These are just defaults. The script that calls things, `mnist-sghmc.py`
        will override.
        """
        # network type
        self.net_type = 'mlp2'
        self.node_type = 'sigmoid'
        self.out_type  = 'softmax'
        #------------------------------------
        # learning rate
        self.eta = 0.01
        # momentum decay
        self.mdecay = 0.1
        # weight decay, 
        self.wd = 0.0
        # number of burn-in round, start averaging after num_burn round
        self.num_burn = 1000
        # mini-batch size used in training
        self.batch_size = 500
        # initial gaussian standard deviation used in weight init
        self.init_sigma = 0.001
        # random number seed
        self.seed = 0
        # weight updating method
        self.updater = 'sgd'
        # temperature: temp=0 means no noise during sampling(MAP inference)
        self.temp = 1.0
        # start sampling weight after this round
        self.start_sample = 1
        #----------------------------------
        # hyper parameter sampling
        self.hyperupdater = 'none'
        # when to start sample hyper parameter
        self.start_hsample = 1
        # Gamma(alpha, beta) prior on regularizer
        self.hyper_alpha = 1.0
        self.hyper_beta  = 1.0        
        # sample hyper parameter each gap_hsample over training data
        self.gap_hsample = 1
        #-----------------------------------
        # adaptive learning rate and momentum
        # by default, no need to set these settings
        self.delta_decay = 0.0
        self.start_decay = None
        self.alpha_decay = 1.0
        self.decay_momentum = 0
        self.init_eta = None
        self.init_mdecay = None        
        #-----------------------
        # following things are not set by user
        # sample weight
        self.wsample = 1.0        
        # round counter
        self.rcounter = 0       


    def gap_hcounter( self ):
        """ How many steps before resample hyper parameter. 
        
        Daniel: by default it's 100 since that means the end of each epoch.
        """
        return int(self.gap_hsample * self.num_train / self.batch_size)


    def adapt_decay( self, rcounter ):
        """ Adapt learning rate and momentum, if necessary.
        
        Daniel: by default, this is never called in the code as start_decay=None.
        """
        # adapt decay ratio
        if self.init_eta == None:
            self.init_eta = self.eta
            self.init_mdecay = self.mdecay
        self.wsample = 1.0
        if self.start_decay == None:
            return

        d_eta = 1.0 * np.power(1.0 + max( rcounter - self.start_decay, 0 ) * self.alpha_decay, 
                                - self.delta_decay )
        assert d_eta - 1.0 < 1e-6 and d_eta > 0.0
        
        if self.decay_momentum != 0:
            d_mom = np.sqrt( d_eta )
            self.wsample = d_mom
        else:
            d_mom = 1.0
            self.wsample = d_eta

        self.eta = d_eta * self.init_eta
        self.mdecay = d_mom * self.init_mdecay
        

    def set_round( self, rcounter ):
        self.rcounter = rcounter
        self.adapt_decay( rcounter )
        if self.updater == 'sgld':
            assert np.abs( self.mdecay - 1.0 ) < 1e-6


    def get_sigma( self ):
        """ Returns the **standard deviation**, not the variance!!

        Daniel: the `var = 2 * eta * mdecay` part is straight from the paper.
        But note the temperature ... AND the division by the entire training
        data size!
        """
        if self.mdecay - 1.0 > -1e-5 or self.updater == 'sgld':
            scale = self.eta / self.num_train
        else:
            scale = self.eta * self.mdecay / self.num_train
        return np.sqrt( 2.0 * self.temp * scale ) 
    

    def need_sample( self ):
        """ 
        Daniel (this assumes default settings): self.start_sample=1 so we're
        never in the first case. The round starts at 0 so for the _first_ epoch
        we do NOT sample, so we don't add Gaussian noise at all. But for all 799
        remaining epochs, rcounter >= 1 obviously so we sample. This might be
        why the first epoch results in fast convergence as we only use gradient
        information, and effectively we do SGD+momentum. (Well, wait, that first
        iteration still has noisy gradients ... but that's fine as that's normal
        SGD+momentum!!)

        Called from the SGHMCUpdater's update method.
        """
        if self.start_sample == None:
            return False
        else:
            return self.rcounter >= self.start_sample


    def need_hsample( self ):
        """ 
        Like with the other samples, we always re-sample hyperparameters after
        the first eopch. Update: don't re-sample if we're using sgd or mom+sgd.
        """
        if self.start_hsample==None or self.updater=='sgd' or self.updater=='momsgd':
            return False
        else:
            return self.rcounter >= self.start_hsample


    def rec_gsqr( self ):
        # whether the network need to provide second moment of gradient
        return False
