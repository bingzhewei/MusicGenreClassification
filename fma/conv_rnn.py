import tensorflow as tf
import numpy as np


def convLSTM_shape(X):
    shape = X.get_shape().as_list()
    nb_shape = [shape[1], shape[3]]
    return nb_shape


def conv_rnn(input_data,
             filters,
             conv_kernels,
             conv_strides,
             lstm_channels,
             lstm_kernels,
             num_classes,
             batch_norm=True,
             reuse=False):
    """
    Network Structure: layers of 2d convolution -> layers of 1d convLSTM -> 1 dense layer to output logits

    @author Zhihan Xiong

    input_data: tensor of input data or placeholder with shape [batch, frequency, time_length, channels]
    filters: an integer list or array to indicate the number of filters in each convolution layer
    conv_kernels: a list of length-2 lists to indicate the kernel size in each convolution layer
    conv_strides: a list of length-2 lists to indicate the stride in each convolution layer
    lstm_channels: an integer list or array to indicate the number of filters in each convLSTM layer
    lstm_kernels: a list of length-1 lists to indicate the kernel size in each convLSTM layer
    num_classes: an integer to indicate the number of classes for classification
    batch_norm: a boolean to indicate if batch normalization is performed after convolution
    reuse: boolean, indicating if the variable will be reused
    """
    assert len(filters) == len(conv_kernels) == len(conv_strides)
    assert len(lstm_channels) == len(lstm_kernels)

    activa = input_data
    num_conv = len(filters)
    for i in range(num_conv):
        with tf.variable_scope("conv_{}".format(i + 1), reuse=reuse):
            conv = tf.layers.conv2d(
                activa,
                filters[i],
                conv_kernels[i],
                conv_strides[i],
                padding='SAME')
            if batch_norm:
                conv = tf.layers.batch_normalization(conv)
            activa = tf.nn.relu(conv)

    conv_out_shape = activa.get_shape().as_list()
    CLSTM_shape = convLSTM_shape(activa)
    split_list = [1 for _ in range(conv_out_shape[2])]
    sequences = tf.split(activa, split_list, axis=2)
    # outputs = [tf.squeeze(seq, axis = [2]) for seq in sequences]
    outputs = tf.stack([tf.squeeze(seq, axis=[2]) for seq in sequences])

    num_lstm = len(lstm_channels)
    for j in range(num_lstm):
        with tf.variable_scope("convlstm_{}".format(j + 1), reuse=reuse):
            CLSTMcell = tf.contrib.rnn.ConvLSTMCell(
                1, CLSTM_shape, lstm_channels[j], lstm_kernels[j])
            # outputs, states = tf.nn.static_rnn(CLSTMcell, outputs, dtype = tf.float32)
            outputs, stats = tf.nn.dynamic_rnn(
                CLSTMcell, outputs, dtype=tf.float32, time_major=True)

    # rnn_out = tf.reshape(outputs[-1], [conv_out_shape[0], -1])
    rnn_out = tf.reshape(outputs[-1],
                         [-1, conv_out_shape[1] * lstm_channels[-1]])
    logits = tf.layers.dense(rnn_out, num_classes, tf.nn.relu)
    return logits


def test_conv_rnn():
    trial_input = tf.placeholder(tf.float32, [None, 100, 50, 2])
    # feed = tf.constant(0, tf.float32, [20, 100, 50, 2])
    feed = np.zeros([20, 100, 50, 2])
    output = conv_rnn(trial_input, [64, 128], [[3, 3], [5, 5]],
                      [[1, 1], [1, 1]], [64, 128], [[3], [5]], 10)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict={trial_input: feed})
    print(result.shape)


if __name__ == '__main__':
    test_conv_rnn()
