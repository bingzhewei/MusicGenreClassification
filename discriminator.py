import tensorflow as tf
from tensorflow.contrib.training import batch_sequences_with_states
from SR_like_AutoEncoder import AutoEncoder

def discriminator(input, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            return discriminator_0(input, reuse)


def discriminator_0(input, reuse):
    auto_encoder = AutoEncoder(input, [256, 512], [3, 3], [2, 1], 1, [1, 1], reuse=reuse)
    return tf.nn.sigmoid(tf.reduce_mean(auto_encoder, axis=[1, 2, 3]))


def discriminator_1(input,
                    reuse,
                    name,
                    conv_filter_out=1024,
                    conv_kernel_size=5,
                    conv_stride=1,
                    lstm_hidden_size=512,
                    dense_shape=[512, 512, 1]):
    """
    @author Fanbo

    input: a tensor with shape [batch_size, n_time_steps]
    reuse: whether we should reuse the variables
    name: name of the generator
    conv_filter_out: filte size for convolution
    conv_kernel_size: kernel size for convolution
    conv_stride: stride size for convolution, kernel size should be a integer multiple of conv_stride
    lstm_hidden_size: hidden states for lstm
    """
    assert conv_kernel_size / conv_stride == conv_kernel_size // conv_stride


    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('discriminator_conv_1', reuse=reuse):
            c1 = tf.layers.conv1d(
                input,
                conv_filter_out,
                conv_kernel_size,
                conv_stride,
                'SAME',
                reuse=reuse)
        with tf.variable_scope('discriminator_lstm_1', reuse=reuse):
            cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, reuse=reuse)
            output, state = tf.nn.static_rnn(cell, tf.unstack(c1, 10, axis=1), dtype=tf.float32)

        stacked_state = tf.concat(state, axis=1)
        current_state = stacked_state
        for i, length in enumerate(dense_shape):
            with tf.variable_scope('discriminator_dense_{}'.format(i + 1), reuse=reuse):
                current_state = tf.layers.dense(
                    current_state, length, tf.nn.relu
                    if i != len(dense_shape) - 1 else None, reuse=reuse)
        output = current_state
        return tf.reshape(output, [tf.shape(input)[0]])


import numpy as np


def _test_discriminator_1():
    input = tf.placeholder(tf.float32, [4, 1000, 1])
    output = discriminator_1(input, False, 'test_1')

    feed = np.zeros([4, 1000, 1])
    sess = tf.Session()
    sess.run([tf.global_variables_initializer()])
    output = result = sess.run([output], feed_dict={input: feed})
    print(output)

if __name__ == '__main__':
    _test_discriminator_1()
