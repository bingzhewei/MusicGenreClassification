import numpy as np
import tensorflow as tf


def all_conv(X,
             filters,
             kernels,
             strides,
             pool_size,
             keep_prob,
             num_classes,
             batch_norm=True,
             reuse=False):
    """
    X: input data of shape [batch, frequency, time, channel]
    """
    num_convs = len(filters)
    assert len(filters) == len(kernels) == len(strides) == len(keep_prob)

    activa = X
    for i in range(num_convs):
        with tf.variable_scope("conv_{}".format(i + 1), reuse=reuse):
            conv = tf.layers.conv2d(
                activa, filters[i], kernels[i], strides[i], padding='SAME')
            pooling = tf.layers.max_pooling2d(conv, pool_size[i], pool_size[i])
            if batch_norm:
                pooling = tf.layers.batch_normalization(pooling)
            drop = tf.nn.dropout(pooling, keep_prob[i])
            activa = tf.nn.relu(drop)
    ac_shape = activa.get_shape().as_list()
    conv_out = tf.reshape(activa,
                          [-1, ac_shape[1] * ac_shape[2] * ac_shape[3]])
    logits = tf.layers.dense(conv_out, num_classes, tf.sigmoid)

    return logits


def test_all_conv():
    trial_input = tf.placeholder(tf.float32, [None, 20, 1254, 2])
    fake_X = np.zeros([20, 20, 1254, 2])
    output = all_conv(trial_input, [64, 128, 256], [[3, 3], [3, 3], [3, 3]],
                      [[1, 1], [1, 1], [1, 1]], [[2, 4], [2, 4],
                                                 [2, 4]], [1.0, 1.0, 1.0], 7)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict={trial_input: fake_X})
    print(result.shape)
