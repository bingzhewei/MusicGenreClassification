import tensorflow as tf

from AudioCNN import LayerUtils as LayerUtils


def FourxFourSegNet(features, reuse=False):
    tf.layers.MaxPooling2D(2, )
    conv1 = LayerUtils.conv2d_layer(features, 3, "VALID", 64, "conv1", reuse)
    conv2 = LayerUtils.conv2d_layer(conv1, 3, "VALID", 64, "conv2", reuse)
    conv3 = LayerUtils.conv2d_layer(conv2, 3, "VALID", 128, "conv3", reuse)
    conv4 = LayerUtils.conv2d_layer(conv3, 3, "VALID", 128, "conv4", reuse)
    conv5 = LayerUtils.conv2d_layer(conv4, 3, "VALID", 256, "conv5", reuse)
    conv6 = LayerUtils.conv2d_layer(conv5, 3, "VALID", 256, "conv6", reuse)
    conv7 = LayerUtils.conv2d_layer(conv6, 3, "VALID", 256, "conv7", reuse)
    conv8 = LayerUtils.conv2d_layer(conv7, 3, "VALID", 256, "conv8", reuse)
    conv9 = LayerUtils.conv2d_layer(conv8, 3, "VALID", 512, "conv9", reuse)
    conv10 = LayerUtils.conv2d_layer(conv9, 3, "VALID", 512, "conv10", reuse)
    conv11 = LayerUtils.conv2d_layer(conv10, 3, "VALID", 512, "conv11", reuse)
    conv12 = LayerUtils.conv2d_layer(conv11, 3, "VALID", 512, "conv12", reuse)

    bn = tf.contrib.layers.batch_norm(
        input_layer, fused=True, data_format='NCHW'
    scope = scope)

    return conv12
