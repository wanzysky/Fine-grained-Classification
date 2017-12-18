from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets.spatial_transformer import transformer
import numpy as np

slim = tf.contrib.slim

W = 7
H = 7
C = 2208
N = 4


@slim.add_arg_scope
def _M(net, d, scope=None, outputs_collections=None):
    """
    :param net: [batch_size, W, H, C]
    :param d:   [batch_size, C, N]
    :return: N * [batch_size, W, H]
    """
    with tf.variable_scope(scope, 'fully_layers', [net]) as scope:
        # net: [batch_size, W, H, C]
        # with shape [batch_size, C, N]
        batch_size = tf.shape(net)[0]
        M = []
        for d_i in d:
            M_i = tf.sigmoid(
                    tf.reduce_sum(
                        tf.multiply(
                            net,
                            tf.reshape(d_i, (batch_size, 1, 1, C))),
                        axis=3))
            M_i = slim.utils.collect_named_outputs(outputs_collections, scope.name, M_i)
            M.append(M_i)
        return M


@slim.add_arg_scope
def _d(net, weight_decay=1e-4, scope=None, outputs_collections=None):
    """
    :param net: [batch_size, W, H, C]
    :return: N * [batch_size, C]
    """
    with tf.variable_scope(scope, 'fully_connected_layers', [net]) as scope:
        # with shape [batch_size, C, N]
        batch_size = tf.shape(net)[0]

        d = []
        for _ in range(N):
            d_i = slim.fully_connected(tf.reshape(net, (batch_size, W * H * C)), C,
                                     activation_fn=None,
                                     weights_regularizer=slim.l2_regularizer(weight_decay))
            d_i = slim.utils.collect_named_outputs(outputs_collections, scope.name, d_i)
            d.append(d_i)
        return d


@slim.add_arg_scope
def _P(net, M, weight_decay=1e-4, scope=None, outputs_collections=None):
    P = []
    for M_i in M:
        with tf.variable_scope(scope, 'P_x', M + net):
            P_i = tf.multiply(tf.reshape(M_i, (-1, W, H, 1)), net)
            P.append(P_i)
    return P


def multiattention(net, initializing=False, weight_decay=1e-4, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'multi_attention', [net]) as scope:
        # with shape [batch_size, C, N]
        d = _d(net, weight_decay=1e-4, scope=scope, outputs_collections=outputs_collections)
        if initializing:
            return _initializing_loss(d)
        M = _M(net, d, scope, outputs_collections)
        # with shape [batch_size, N, W, H, C]
        P = []
        for M_i in M:
            P_i = tf.multiply(net, tf.reshape(M_i, (-1, W, H, 1)))
            P.append(P_i)
        return P


def _channel_group_loss(net, M, margin, lamda, scope=None, outputs_collections=None):
    return _distance_loss(net, M, scope, outputs_collections) + \
            _diversity_loss(M, margin, lamda, scope, outputs_collections)


def _distance_loss(net, M, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'distance_loss', [M + net]) as sc:
        batch_size = tf.shape(net)[0]
        xs = tf.reshape(
                tf.tile(
                    tf.range(W, dtype=tf.float32),
                    [C * H * batch_size]),
                (batch_size, W, H, C))
        xs = xs / W
        ys = tf.reshape(
                tf.tile(
                    tf.range(H, dtype=tf.float32),
                    [C * W * batch_size]),
                (batch_size, W, H, C))
        ys = tf.transpose(ys, (0, 2, 1, 3))
        ys = ys / H

        tx = tf.argmax(
                tf.reduce_max(
                    net,
                    axis=2,
                    keep_dims=True),
                axis=1)
        tx = tf.reshape(tx, (batch_size, 1, 1, C))
        tx = tf.cast(tx, 'float32')
        tx = tf.tile(tx, [1, W, H, 1])
        ty = tf.argmax(
                tf.reduce_max(
                    net,
                    axis=1,
                    keep_dims=True),
                axis=2)
        ty = tf.cast(ty, 'float32')
        ty = tf.reshape(ty, (batch_size, 1, 1, C))
        ty = tf.tile(ty, [1, W, H, 1])

        part_dis_losses = []
        for M_i in M:
            with tf.variable_scope(sc, 'part_x', [net, M_i]) as part_sc:
                m_i = tf.reshape(M_i, (batch_size, W, H, 1))
                part_dis_loss = tf.reduce_sum(
                        tf.multiply(
                            M_i,
                            tf.square(xs - tx) + tf.square(ys - ty)))
                part_dis_losses.append(part_dis_loss)
                tf.losses.add_loss(part_dis_loss)
        return part_dis_losses


def _diversity_loss(M, margin, lamda, scope=None, outputs_collections=None):
    with tf.variable_scope(scope, 'diversity_loss', M) as sc:
        div_losses = []
        for i in range(N):
            # [batch_size * (N - 1) * W * H] ==> [batch_size * W * H]
            exclusive_max = tf.reduce_max(
                    tf.concat(M[:i] + M[i:], axis=1),
                    axis=1,
                    keep_dims=False)
            with tf.variable_scope(sc, 'item_x', [exclusive_max]):
                i_th_loss = tf.reduce_sum(
                        tf.multiply(
                            M[i],
                            exclusive_max - margin))
                i_th_loss = i_th_loss * lamda
                tf.losses.add_loss(i_th_loss)

                div_losses.append(i_th_loss)
        return div_losses


def loss(net, M, margin=0.02, lamda=2, initializing=False, scope=None, outputs_collections=None):
    return _channel_group_loss(net, M, margin, lamda, scope, outputs_collections)


def _init_label(i):
    init_label = np.zeros((C), dtype=np.float32)
    split_size, remainder = divmod(C, N)
    if remainder > 0:
        split_size += 1
    init_label[i*split_size:(i+1)*split_size] = 1

    return init_label


def _initializing_loss(d, batch_size, scope=None, outputs_collections=None):
    losses = []
    with tf.variable_scope(scope, 'initializing_loss', d) as sc:
        for i in range(N):
            with tf.variable_scope(sc, 'loss_x', [d[i]]):
                init_label = tf.constant(_init_label(i), dtype=tf.float32)
                label = tf.reshape(tf.tile(init_label, batch_size), (batch_size, C))
                loss = tf.losses.softmax_cross_entropy(d[i], label)
                tf.losses.add_loss(loss)
                losses.append(loss)
    return losses


