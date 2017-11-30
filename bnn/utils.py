""" 
To reduce clutter.  BTW: at some point I need my own personal library of this
rather than copying and pasting.
"""
import gzip, pickle, os, sys
import numpy as np
import tensorflow as tf
from collections import defaultdict


# ------------------------------------------------------------------------------
# TF Sessions and running simple tests
# ------------------------------------------------------------------------------

def get_tf_session(gpumem=0.75):
    """ Returning a session. Set options here if desired. """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpumem)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def train_normal(mnist, sess):
    """ Normal mnist, just to get my feet wet. """
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess.run(tf.global_variables_initializer())
    
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# ------------------------------------------------------------------------------
# Scopes and weights
# ------------------------------------------------------------------------------

def scope_vars(scope, trainable_only=False):
    """ Get variables inside a scope (typically a string); from OpenAI.

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    trainable_only: bool
        whether or not to return only the variables that were marked as
        trainable.

    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def vars_to_vector(weights):
    """ 
    Given `weights`, a list of TF vars (e.g. from `scope_vars`) return the
    concatenation into a vector. 
    """
    return tf.concat([tf.reshape(w, [-1]) for w in weights], axis=0)


def get_num_weights(weights):
    """ Return number of weights (e.g. from `scope_vars`). """
    w_shapes = [w.get_shape().as_list() for w in weights]
    return np.sum([np.prod(sh) for sh in w_shapes])


def set_weights_from_vector(weights, new_weights_v):
    """ Set network weights to be those from `new_weights_v`, a TF placeholder.

    Specifically, this returns the TF operation which we call in a session.
    We also need `weights`, which we can get from `scope_vars`.
    """
    updates = []
    start = 0
    for (i,w) in enumerate(weights):
        shape = w.get_shape().as_list()
        size = np.prod(shape)
        updates.append(
                tf.assign(w, tf.reshape(new_weights_v[start:start+size], shape))
        )
        start += size
    net_set_params = tf.group(*updates)
    return net_set_params


# ------------------------------------------------------------------------------
# TF convenience functions to replicate numpy
# ------------------------------------------------------------------------------

# So we can avoid typing tf.clip_by_value, etc., in other code.
clip = tf.clip_by_value

def fancy_slice_2d(X, inds0, inds1):
    """ Like numpy's X[inds0, inds1]. From OpenAI's code (for CS 294-112). """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def sum(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)

def mean(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)


# ------------------------------------------------------------------------------
# Loading datasets
# ------------------------------------------------------------------------------

def load_dataset(name):
    """ Given dataset, return train, valid, test sets. And length stats.
    
    For MNIST, the pickled data is from Python 2, gah. Fortunately this blog
    helped: http://www.mlblog.net/2016/09/reading-mnist-in-python3.html. The
    train, val, and test are both tuples with the first and second elements as
    the data and labels. Arrange things in a dictionary. BTW it is already
    downscaled apropriately to [0,1].
    """
    if name == 'mnist':
        with gzip.open('../data/mnist.pkl.gz','rb') as ff:
            u = pickle._Unpickler( ff )
            u.encoding = 'latin1'
            train, val, test = u.load()
        data = {}
        data['X_train'] = train[0]
        data['y_train'] = train[1]
        data['X_valid'] = val[0]
        data['y_valid'] = val[1]
        data['X_test'] = test[0]
        data['y_test'] = test[1]
        print("\nWe loaded MNIST. Shapes:")
        print("  X_train.shape: {}".format(data['X_train'].shape))
        print("  y_train.shape: {}".format(data['y_train'].shape))
        print("  X_valid.shape: {}".format(data['X_valid'].shape))
        print("  y_valid.shape: {}".format(data['y_valid'].shape))
        print("  X_test.shape:  {}".format(data['X_test'].shape))
        print("  y_test.shape:  {}".format(data['y_test'].shape))
        assert (0.0 <= np.min(data['X_train']) and np.max(data['X_train']) <= 1.0)
        print("(assertion passed, values are in [0,1])\n")
        stats = {'num_train': len(data['y_train']), 
                 'num_valid': len(data['y_valid']), 
                 'num_test':  len(data['y_test'])}
        return (data, stats)
    else:
        raise ValueError("Dataset name {} is not valid".format(name))


def list_of_minibatches(data, bsize, shuffle=True):
    """ Forms a list of minibatches for each element in `data` to avoid
    repeatedly shuffling and sampling during training.

    Assumes `data` is a dictionary with `X_train` and `y_train` keys, and
    returns a dictionary of the same length.
    """
    data_lists = defaultdict(list)
    assert 'X_train' in data.keys() and 'y_train' in data.keys()
    N = data['X_train'].shape[0]
    indices = np.random.permutation(N)
    X_train = data['X_train'][indices]
    y_train = data['y_train'][indices]

    for i in range(0, N-bsize, bsize):
        data_lists['X_train'].append(X_train[i:i+bsize, :])
        data_lists['y_train'].append(y_train[i:i+bsize])

    first = data_lists['X_train'][0].shape
    last  = data_lists['X_train'][-1].shape
    assert first == last, "{} vs {} w/bs {}".format(first, last, bsize)

    # Now do validation.
    N = data['X_valid'].shape[0]
    indices = np.random.permutation(N)
    X_train = data['X_valid'][indices]
    y_train = data['y_valid'][indices]

    for i in range(0, N-bsize, bsize):
        data_lists['X_valid'].append(X_train[i:i+bsize, :])
        data_lists['y_valid'].append(y_train[i:i+bsize])

    return data_lists
