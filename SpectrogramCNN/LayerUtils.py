import numpy as np
import tensorflow as tf


def conv1d_layer(input, kernel_size, padding, num_filters, mask_type, name, reuse_var=False):
    in_channels = input.get_shape().as_list()[3]

    # assert in_channels >= 0

    with tf.variable_scope(name, reuse=reuse_var) as scope:
        kernel = tf.get_variable("kernel", shape=[kernel_size,
                                                  in_channels, num_filters],
                                 initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))
        mask = np.ones((kernel_size, in_channels, num_filters), np.float32)
        if mask_type == 'above_T':
            mask =
        elif mask_type == 'centered_T':
            mask =
        elif mask_type == 'above_triangle':
            mask =
        elif mask_type == 'centered_triangle':
            mask =

        kernel *= tf.constant(mask, tf.float32)

        conv = tf.nn.conv1d(input, kernel, [1, 1, 1, 1], padding=padding, name="conv2d")
        biases = tf.get_variable("bias", shape=[num_filters],
                                 initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))
        bias = tf.nn.bias_add(conv, biases)
        output = tf.nn.relu(bias, name="output_cropped")
        return output


def fc_layer_relu(input, num_hidden, name, reuse_var=False):
    in_channels = input.get_shape().as_list()[1]

    # assert in_channels >= 0

    with tf.variable_scope(name, reuse=reuse_var) as scope:
        kernel = tf.get_variable("kernel", shape=[in_channels, num_hidden],
                                 initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))
        biases = tf.get_variable("bias", shape=[num_hidden],
                                 initializer=tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG'))
        bias = tf.nn.bias_add(tf.matmul(input, kernel), biases)
        output = tf.nn.relu(bias, name="output_cropped")
        return output


def flatten_layer(input, name, reuse_var=False):
    in_channels = np.prod(input.get_shape().as_list()[1:])

    with tf.variable_scope(name, reuse=reuse_var) as scope:
        output = tf.reshape(input, [-1, in_channels], "flatten")
        return output


def dropout_layer(input, keep_prob, name, reuse_var=False):
    with tf.variable_scope(name, reuse=reuse_var) as scope:
        output = tf.nn.dropout(input, keep_prob, name="dropout")
        return output
