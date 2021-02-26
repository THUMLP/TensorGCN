import tensorflow as tf
from inits import *
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def dot1(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        y = tf.sparse_tensor_to_dense(y)
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.sparse_tensor_dense_matmul(x, y)
    return res


def dot2(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(y, x)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        self.vars_mix = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution_mix1(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.leaky_relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution_mix1, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.support_mix = placeholders['support_mix']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self.input_dim = input_dim
        self.output_dim = output_dim

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support_mix)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
                self.vars['weights_' + str(i) + str(i)] = glorot([output_dim, output_dim],
                                                                 name='weights_' + str(i) + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def atten(self, supports):
        for i in range(len(supports)):
            supports[i] = dot(supports[i], self.vars['weights_' + str(i) + str(i)])
        att_features = []
        att_features.append(tf.nn.leaky_relu(tf.add(supports[1], supports[2])))
        att_features.append(tf.nn.leaky_relu(tf.add(supports[0], supports[2])))
        att_features.append(tf.nn.leaky_relu(tf.add(supports[0], supports[1])))
        return att_features

    def _call(self, inputs):
        xx = inputs
        # dropout
        for i in range(len(xx)):
            if self.sparse_inputs:
                xx[i] = sparse_dropout(xx[i], 1 - self.dropout, self.num_features_nonzero)
            else:
                xx[i] = tf.nn.dropout(xx[i], 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support_mix)):
            if not self.featureless:
                pre_sup = dot(xx[i], self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support_mix[i], pre_sup, sparse=True)
            support = tf.nn.leaky_relu(support)
            supports.append(support)

        supports = self.atten(supports)

        # get embedding
        self.embedding = tf.stack([supports[-3], supports[-2], supports[-1]], axis=0)
        self.embedding = tf.reduce_mean(self.embedding, axis=0)
        return supports
