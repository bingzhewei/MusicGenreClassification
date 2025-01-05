import tensorflow as tf
from SR_like_AutoEncoder import AutoEncoder

num_hidden_units = 1024
num_layers = 3


def generator(input, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('generator', reuse=reuse):
            # return generator_1(input, reuse, name)
            return AutoEncoder(input, [256, 512], [3, 3], [2, 1], 1, [1, 1], reuse=reuse)


def generator_1(input,
                reuse,
                name,
                conv_filter_size=1024,
                conv_kernel_size=5,
                conv_stride=1,
                lstm_hidden_size=512):
    """
    @author Fanbo
    +-------+    +-----+    +------+    +-------------+    +--------+
    | input | -> | CNN | -> | LSTM | -> | inverse CNN | -> | output |
    +-------+    +-----+    +------+    +-------------+    +--------+

    input: a tensor with shape [batch_size, n_time_steps]
    reuse: whether we should reuse the variables
    name: name of the generator
    conv_filter_size: filte size for convolution
    conv_kernel_size: kernel size for convolution
    conv_stride: stride size for convolution, kernel size should be a integer multiple of conv_stride
    lstm_hidden_size: hidden states for lstm
    """
    assert conv_kernel_size / conv_stride == conv_kernel_size // conv_stride

    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('generator_conv_1', reuse=reuse):
            c1 = tf.layers.conv1d(input, conv_filter_size, conv_kernel_size,
                                  conv_stride, 'SAME', reuse=reuse)
        with tf.variable_scope('generator_lstm_1', reuse=reuse):
            cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size, reuse=reuse)
            output, state = tf.nn.static_rnn(cell, tf.unstack(c1, 10, axis=1), dtype=tf.float32)
            output = tf.stack(output, axis=1)
        with tf.variable_scope('generator_conv_2', reuse=reuse):
            c2 = tf.layers.conv1d(output, 1, conv_kernel_size, conv_stride, 'SAME', reuse=reuse)
        return c2


import numpy as np


def _test_1():
    dim = [4, 1000, 1]

    input = tf.placeholder(tf.float32, dim)
    output = generator_1(input, False, 'test_1')

    feed = np.zeros(dim)
    sess = tf.Session()
    sess.run([tf.global_variables_initializer()])
    result = sess.run([output], feed_dict={input: feed})
    assert result[0].shape == feed.shape

    # test shared parameters
    input2 = tf.placeholder(tf.float32, dim)
    output2 = generator_1(input2, True, 'test_1')

    input3 = tf.placeholder(tf.float32, dim)
    output3 = generator_1(input3, False, 'test_2')

    try:
        # This should fail
        input4 = tf.placeholder(tf.float32, dim)
        output3 = generator_1(input3, False, 'test_1')
    except:
        pass
    else:
        print("This should fail but passes")
        assert False

    print('test_1 finished')


if __name__ == '__main__':
    _test_1()
