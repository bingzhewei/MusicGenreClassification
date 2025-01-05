import tensorflow as tf
from conv_rnn import conv_rnn
import mfcc_reader
import numpy as np

total_size = 3000
batch_size = 30
if_batch_norm = True

batch_creator = mfcc_reader.Loader()

with tf.Session() as sess:
    batch_creator.start_threads(sess, False)
    X_batch, y_batch = batch_creator.dequeue(batch_size)
    X_batch = tf.reshape(X_batch, X_batch.get_shape().as_list() + [1])
    y_rnn = conv_rnn(X_batch, [# 16, 32
    ], [# [3, 3], [5, 5]
    ], [# [1, 1], [1, 1]
    ],
                     [16, 32], [[3], [5]], 7, if_batch_norm)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "conv_rnn/")

    preds = tf.argmax(y_rnn, 1)
    true_lab = tf.argmax(y_batch, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, true_lab), tf.float32))

    test_acc = []
    for i in range(int(total_size/batch_size)):
        test_acc.append(sess.run(accuracy))
    total_acc = np.mean(np.array(test_acc))
    print("The total accuracy is {}".format(total_acc))
