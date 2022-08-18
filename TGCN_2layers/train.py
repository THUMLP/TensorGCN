from __future__ import division
from __future__ import print_function

import os
import random
import time

import tensorflow as tf
from models import GCN
from sklearn import metrics
from utils import *

dataset = 'mr'
os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
os.path.abspath(os.path.dirname(os.getcwd()))
os.path.abspath(os.path.join(os.getcwd(), ".."))
f_file = os.sep.join(['..', 'data_tgcn', dataset, 'build_train'])

# Set random seed
#seed = random.randint(1, 200)
seed=148
print(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('num_labels', 2, 'Number of units in mlp2 output1.')
flags.DEFINE_float('dropout', 0.9, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.000001,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 2000,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, adj1, adj2, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size, test_size = load_corpus(
    f_file, FLAGS.dataset)
print(adj)
# print(adj[0], adj[1])
features = sp.identity(features.shape[0])  # featureless

print(adj.shape)
print(features.shape)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    ##original
    # support = [preprocess_adj(adj)]
    # num_supports = 1
    # modify for mix
    support = [preprocess_adj(adj)]
    num_supports = 1
    support_mix = [preprocess_adj_mix(adj), preprocess_adj_mix(adj1), preprocess_adj_mix(adj2)]
    num_supports_mix = 3
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'support_mix': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports_mix)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
print(features[2][1])
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, support_mix, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

istrain = True
if istrain:
    cost_valid = []
    acc_valid = []
    max_acc = 0.0
    min_cost = 10.0
    with open("record_{}.txt".format(FLAGS.dropout),'w',encoding='utf8') as f:
        print(str(FLAGS.dropout))
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(
                features, support, support_mix, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

            # Validation
            valid_cost, valid_acc, pred, labels, duration = evaluate(
                features, support, y_val, val_mask, placeholders)

            # Testing
            test_cost, test_acc, pred, labels, test_duration = evaluate(
                features, support, y_test, test_mask, placeholders)

            cost_valid.append(valid_cost)
            acc_valid.append(valid_acc)
            tmp = "Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "train_acc=","{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(valid_cost),"val_acc=", "{:.5f}".format(valid_acc), "test_loss=", "{:.5f}".format(test_cost), "test_acc=","{:.5f}".format(test_acc), "time=", "{:.5f}".format(time.time() - t)
            print(tmp)
            f.write(' '.join(tmp))
            f.write('\n')
            # save model
            if epoch >900 and  cost_valid[-1] < min_cost:
                model.save(sess)
                min_cost = cost_valid[-1]

            if epoch > FLAGS.early_stopping and cost_valid[-1] > np.mean(cost_valid[-(FLAGS.early_stopping + 1):-1]):
                print("Early stopping...")
                break

        print("Optimization Finished!")


else:
    model.load(sess)
    # Testing
    test_cost, test_acc, pred, labels, test_duration = evaluate(
        features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    test_pred = []
    test_labels = []
    print(len(test_mask))
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(labels[i])

    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_labels, test_pred, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))