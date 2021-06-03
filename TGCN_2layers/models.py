import tensorflow as tf
from layers import *
from metrics import *
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        self.activations.append(self.inputs)
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-3:])
            self.activations.extend(hidden)
        self.outputs = tf.stack([self.activations[-3], self.activations[-2], self.activations[-1]], axis=0)
        self.outputs = tf.reduce_mean(self.outputs, axis=0)

        # tmp = tf.concat([self.activations[3],self.activations[4],self.activations[5],self.activations[6],self.activations[7],self.activations[8]], -1)
        # with tf.variable_scope(self.name + '_vars'):
        #     while tmp.shape[1].value > self.output_dim * 2:
        #         weight = glorot([tmp.shape[1].value, tmp.shape[1].value // 2])
        #         tmp = tf.nn.tanh(tf.matmul(tmp, weight))
        #     weight = glorot([tmp.shape[1].value, self.output_dim])
        #     tmp = tf.matmul(tmp, weight)
        # self.outputs = tmp

        # Store model variables for easy access
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess,
                               "../data_tgcn/mr/build_train/{}_best_models/%s.ckpt".format(FLAGS.dataset) % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "../data_tgcn/mr/build_train/{}_best_models/%s.ckpt".format(FLAGS.dataset) % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self):
        self.layers.append(GraphConvolution_mix1(input_dim=self.input_dim,
                                                 output_dim=FLAGS.hidden1,
                                                 placeholders=self.placeholders,
                                                 act=tf.nn.leaky_relu,
                                                 dropout=True,
                                                 featureless=False,
                                                 sparse_inputs=True,
                                                 logging=self.logging))

        self.layers.append(GraphConvolution_mix1(input_dim=FLAGS.hidden1,
                                                 output_dim=self.output_dim,
                                                 placeholders=self.placeholders,
                                                 act=lambda x: x,  #
                                                 dropout=True,
                                                 featureless=False,
                                                 sparse_inputs=False,
                                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
