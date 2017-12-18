from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets.spatial_transformer import transformer
import numpy as np

slim = tf.contrib.slim

W = 7
H = 7
C = 96
N = 4

@slim.add_arg_scope
def _M(net, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'fully_layers', [net]) as scope:
        # net: [batch_size, W, H, C]
        # with shape [batch_size, C, N]
        batch_size = tf.shape(net)[0]
        M = slim.fully_connected(tf.reshape(net, (batch_size, W*C*H)), N,
                activation_fn=tf.sigmoid,
                weights_regularizer=slim.l2_regularizer)
        M = slim.utils.collect_named_outputs(outputs_collections, scope.name, M)
        return M


def multi_attention(net, N, initializing=False, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'multi_attention', [net]) as scope:
        # with shape [batch_size, C, N]
        M = _M(net, N)
        if initializing:
            batch_size = tf.shape(net)[0]
            labels = tf.zeros((batch_size, C, N))

            tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=label, weights=0.5)
        # with shape [batch_size, N, W, H, C]
        P = tf.multiply(net, M)
        return P



