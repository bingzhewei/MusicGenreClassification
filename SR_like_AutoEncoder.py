import tensorflow as tf
import numpy as np


def onedim_2_twodim(tensor):
    """
    Reshape a 1D data tensor with shape (batch, length, channels) to a 2D data tensor
    with shape (batch, 1, length, channels)

    tensor: the input tensor
    """
    shape = tensor.get_shape().as_list()
    twod_data = tf.reshape(tensor, [shape[0], 1, -1, shape[2]])
    return twod_data


def twodim_2_onedim(tensor):
    """
    The reverse of onedim_2_twodim
    """
    shape = tensor.get_shape().as_list()
    oned_data = tf.reshape(tensor, [shape[0], -1, shape[3]])
    return oned_data


def AutoEncoder(input_data,
                filters,
                kernel_size,
                strides,
                num_of_dense_layers,
                keep_prob,
                num_of_neurons=None,
                residual=False,
                reuse=False):
    """
    @author Zhihan

    input_data: tensor of input data or placeholder with shape (batch, 1, length, channels)
    filters: an integer list or array to indicate the number of filters in each convolution layer
    kernel_size: indicate the size of kernel in each convolution layer
    strides: indicate the strides in each convolution layer
    num_of_dense_layers: integer, indicate the number of fully connected layers in the middle
    num_of_neurons: an integer list or array to indicate the number of neurons in each dense layer
    keep_prob: an integer list or array to indicate the keep probability in each upsampling layer
    residual: boolean, indicating if the residual connection between downsampling and upsampling will be used
    reuse: boolean, indicating if the variable will be reused
    """
    num_of_Dblock = len(filters)
    assert len(filters) == len(kernel_size) == len(strides)
    if num_of_neurons is not None:
        assert num_of_dense_layers == len(num_of_neurons)
        assert num_of_neurons[0] == num_of_neurons[-1]

    input_shape = input_data.get_shape().as_list()
    activa = input_data
    downsample_results = []
    for i in range(num_of_Dblock):
        with tf.variable_scope("D_block_{}".format(i + 1), reuse=reuse):
            conv = tf.layers.conv2d(activa, filters[i], [1, kernel_size[i]],
                                    (1, strides[i]), padding='SAME')
            batch_norm = tf.layers.batch_normalization(conv)
            activa = tf.nn.relu(batch_norm)
        downsample_results.append(activa)

    # twod_tensor = tf.reshape(activa, [input_shape[0], -1])
    fourd_audio = activa
    # activa_shape = activa.get_shape().as_list()
    # twod_tensor = tf.reshape(activa, [-1, activa_shape[2] * activa_shape[3]])
    # twod_shape = twod_tensor.get_shape().as_list()
    # if num_of_neurons is None:
    #     num_of_neurons = [twod_shape[1] for i in range(num_of_dense_layers)]
    # for j in range(num_of_dense_layers):
    #     with tf.variable_scope("B_block_{}".format(j + 1), reuse=reuse):
    #         twod_tensor = tf.layers.dense(
    #             twod_tensor, num_of_neurons[j], activation=tf.nn.relu)
    #
    # # fourd_audio = tf.reshape(twod_tensor, [input_shape[0], 1, -1, filters[num_of_Dblock-1]])
    # fourd_audio = tf.reshape(twod_tensor,
    #                          [-1, 1, activa_shape[2], activa_shape[3]])
    for k in range(num_of_Dblock - 1):
        with tf.variable_scope("U_block_{}".format(k + 1), reuse=reuse):
            if residual:
                fourd_audio = fourd_audio + downsample_results[num_of_Dblock
                                                               - k - 1]
            deconv = tf.layers.conv2d_transpose(
                fourd_audio, filters[::-1][k + 1], [1, kernel_size[::-1][k]],
                (1, strides[::-1][k]), padding='SAME')
            drop = tf.nn.dropout(deconv, keep_prob=keep_prob[k])
            fourd_audio = tf.nn.relu(drop)
    with tf.variable_scope("U_block_{}".format(num_of_Dblock), reuse=reuse):
        if residual:
            fourd_audio = fourd_audio + downsample_results[0]
        final_out = tf.layers.conv2d_transpose(
            fourd_audio,
            input_shape[3], [1, kernel_size[0]], (1, strides[0]),
            activation=tf.nn.relu, padding='SAME')

    return final_out


def test_autoencoder():
    trial_input = tf.placeholder(tf.float32, [None, 1, 1000, 2])
    output = AutoEncoder(trial_input, [256, 512], [3, 3], [2, 1], 1, [0.7, 0.7])

    feed = np.ones([20, 1, 1000, 2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict={trial_input: feed})
    print(result.shape)


if __name__ == '__main__':
    test_autoencoder()
