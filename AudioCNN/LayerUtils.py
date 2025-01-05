import math

import numpy as np
import tensorflow as tf


def conv1d_layer(input_tensor, kernel_size, padding, num_filters, stride=1, atrous_function=None, name="conv1d_layer",
                 reuse_var=False):
    in_channels = input_tensor.get_shape().as_list()[2]

    assert in_channels > 0
    assert num_filters > 0
    assert kernel_size > 0
    assert kernel_size % 2 == 1

    with tf.variable_scope(name, reuse=reuse_var):
        if atrous_function is None or kernel_size == 1:
            kernel = tf.get_variable("kernel", shape=[kernel_size,
                                                      in_channels, num_filters],
                                     initializer=tf.variance_scaling_initializer(mode='fan_in'))
        else:
            atrous_func_inputs = np.arange(1, int(math.ceil(kernel_size / 2)))
            atrous_spacings = np.fromiter((atrous_function(xi) for xi in atrous_func_inputs), atrous_func_inputs.dtype)
            atrous_ones_locations = np.cumsum(atrous_spacings + 1)

            kernel_size_with_atrous = 1 + 2 * (atrous_ones_locations[-1])

            kernel = tf.get_variable("kernel", shape=[kernel_size_with_atrous,
                                                      in_channels, num_filters],
                                     initializer=tf.variance_scaling_initializer(mode='fan_in'))
            mask = np.zeros((kernel_size_with_atrous, in_channels, num_filters), np.float32)

            mask[int(math.floor(kernel_size_with_atrous / 2))] = 1
            mask[int(math.floor(kernel_size_with_atrous / 2)) + atrous_ones_locations, :, :] = 1
            mask[int(math.floor(kernel_size_with_atrous / 2)) - atrous_ones_locations, :, :] = 1

            # if mask_type == 'causal':
            #     mask[:np.floor(kernel_size/2), :, :] = 0
            # elif mask_type == 'anticausal':
            #     mask[np.ceil(kernel_size/2):, :, :] = 0

            kernel *= tf.constant(mask, tf.float32, name="mask")

        conv = tf.nn.conv1d(input_tensor, kernel, stride=stride, padding=padding, use_cudnn_on_gpu=True,
                            data_format="NHWC", name="conv1d")
        biases = tf.get_variable("bias", shape=[num_filters],
                                 initializer=tf.variance_scaling_initializer(mode='fan_in'))
        bias = tf.nn.bias_add(conv, biases, data_format="NHWC", name="bias_add")
        output = tf.nn.relu(bias, name="relu")
        return output
