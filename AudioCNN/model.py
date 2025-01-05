import tensorflow as tf

from AudioCNN import LayerUtils as LayerUtils


def NetGPU0(features, reuse=False):
    conv1 = LayerUtils.conv1d_layer(features, 5, "SAME", 64, 1, lambda x: x ** 2, "conv1", reuse)
    conv2 = LayerUtils.conv1d_layer(conv1, 5, "SAME", 64, 1, lambda x: x ** 2, "conv2", reuse)
    conv3 = LayerUtils.conv1d_layer(conv2, 5, "SAME", 128, 1, lambda x: x ** 2, "conv3", reuse)
    conv4 = LayerUtils.conv1d_layer(conv3, 5, "SAME", 128, 1, lambda x: x ** 2, "conv4", reuse)
    maxpool1 = tf.layers.average_pooling1d(conv4, 9, 3, "valid", data_format="channels_last", name="pool1")
    conv5 = LayerUtils.conv1d_layer(maxpool1, 5, "SAME", 256, 1, lambda x: x ** 2, "conv5", reuse)
    conv6 = LayerUtils.conv1d_layer(conv5, 5, "SAME", 256, 1, lambda x: x ** 2, "conv6", reuse)
    conv7 = LayerUtils.conv1d_layer(conv6, 5, "SAME", 256, 1, lambda x: x ** 2, "conv7", reuse)
    conv8 = LayerUtils.conv1d_layer(conv7, 5, "SAME", 256, 1, lambda x: x ** 2, "conv8", reuse)
    maxpool2 = tf.layers.average_pooling1d(conv8, 9, 3, "valid", data_format="channels_last", name="pool2")
    conv9 = LayerUtils.conv1d_layer(maxpool2, 5, "SAME", 512, 1, lambda x: x ** 2, "conv9", reuse)
    conv10 = LayerUtils.conv1d_layer(conv9, 5, "SAME", 512, 1, lambda x: x ** 2, "conv10", reuse)
    conv11 = LayerUtils.conv1d_layer(conv10, 5, "SAME", 512, 1, lambda x: x ** 2, "conv11", reuse)
    conv12 = LayerUtils.conv1d_layer(conv11, 5, "SAME", 512, 1, lambda x: x ** 2, "conv12", reuse)
    maxpool3 = tf.layers.average_pooling1d(conv12, 9, 3, "valid", data_format="channels_last", name="pool3")
    conv13 = LayerUtils.conv1d_layer(maxpool3, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv13", reuse)
    conv14 = LayerUtils.conv1d_layer(conv13, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv14", reuse)
    conv15 = LayerUtils.conv1d_layer(conv14, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv15", reuse)
    conv16 = LayerUtils.conv1d_layer(conv15, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv16", reuse)
    conv17 = LayerUtils.conv1d_layer(conv16, 3, "SAME", 163, 1, lambda x: x ** 2, "conv17", reuse)

    gap = tf.layers.average_pooling1d(conv17, 4080, 1, "valid", data_format="channels_last", name="gap")
    gap = tf.squeeze(gap, axis=1, name="gap_squeeze")
    return gap


def NetGPU1(features, reuse=False):
    conv1 = LayerUtils.conv1d_layer(features, 5, "SAME", 64, 1, lambda x: x ** 2, "conv1", reuse)
    conv2 = LayerUtils.conv1d_layer(conv1, 5, "SAME", 64, 1, lambda x: x ** 2, "conv2", reuse)
    conv3 = LayerUtils.conv1d_layer(conv2, 5, "SAME", 128, 1, lambda x: x ** 2, "conv3", reuse)
    conv4 = LayerUtils.conv1d_layer(conv3, 5, "SAME", 128, 1, lambda x: x ** 2, "conv4", reuse)
    maxpool1 = tf.layers.max_pooling1d(conv4, 9, 3, "valid", data_format="channels_last", name="pool1")
    conv5 = LayerUtils.conv1d_layer(maxpool1, 5, "SAME", 256, 1, lambda x: x ** 2, "conv5", reuse)
    conv6 = LayerUtils.conv1d_layer(conv5, 5, "SAME", 256, 1, lambda x: x ** 2, "conv6", reuse)
    conv7 = LayerUtils.conv1d_layer(conv6, 5, "SAME", 256, 1, lambda x: x ** 2, "conv7", reuse)
    conv8 = LayerUtils.conv1d_layer(conv7, 5, "SAME", 256, 1, lambda x: x ** 2, "conv8", reuse)
    maxpool2 = tf.layers.max_pooling1d(conv8, 9, 3, "valid", data_format="channels_last", name="pool2")
    conv9 = LayerUtils.conv1d_layer(maxpool2, 5, "SAME", 512, 1, lambda x: x ** 2, "conv9", reuse)
    conv10 = LayerUtils.conv1d_layer(conv9, 5, "SAME", 512, 1, lambda x: x ** 2, "conv10", reuse)
    conv11 = LayerUtils.conv1d_layer(conv10, 5, "SAME", 512, 1, lambda x: x ** 2, "conv11", reuse)
    conv12 = LayerUtils.conv1d_layer(conv11, 5, "SAME", 512, 1, lambda x: x ** 2, "conv12", reuse)
    maxpool3 = tf.layers.max_pooling1d(conv12, 9, 3, "valid", data_format="channels_last", name="pool3")
    conv13 = LayerUtils.conv1d_layer(maxpool3, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv13", reuse)
    conv14 = LayerUtils.conv1d_layer(conv13, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv14", reuse)
    conv15 = LayerUtils.conv1d_layer(conv14, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv15", reuse)
    conv16 = LayerUtils.conv1d_layer(conv15, 5, "SAME", 1024, 1, lambda x: x ** 2, "conv16", reuse)
    conv17 = LayerUtils.conv1d_layer(conv16, 3, "SAME", 163, 1, lambda x: x ** 2, "conv17", reuse)

    gap = tf.layers.average_pooling1d(conv17, 4080, 1, "valid", data_format="channels_last", name="gap")
    gap = tf.squeeze(gap, axis=1, name="gap_squeeze")
    return gap