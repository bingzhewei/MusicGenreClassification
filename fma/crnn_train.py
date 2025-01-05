import numpy as np
import tensorflow as tf
from conv_rnn import conv_rnn
import mfcc_reader


max_step = 5000
batch_size = 50
learning_rate = 1e-4
if_batch_norm = True
save_interval = 10

batch_creator = mfcc_reader.Loader()

with tf.Session() as sess:
    batch_creator.start_threads(sess)
    X_batch, y_batch = batch_creator.dequeue(batch_size)
    X_batch = tf.reshape(X_batch, X_batch.get_shape().as_list() + [1])
    y_rnn = conv_rnn(X_batch, [# 16, 32
    ], [# [3, 3], [5, 5]
    ], [ # [1, 1], [1, 1]
    ], [16, 32], [[3], [5]], 7, if_batch_norm)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=y_rnn))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    conf_matrix = tf.confusion_matrix(
        tf.argmax(y_batch, 1), tf.argmax(y_rnn, 1), 7)
    correct_prediction = tf.equal(tf.argmax(y_rnn, 1), tf.argmax(y_batch, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    summary_writer = tf.summary.FileWriter("./graphs/conv_rnn", sess.graph)
    saver = tf.train.Saver(max_to_keep=20)

    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)

    summary_op = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for i in range(max_step):
        _, summary = sess.run([train_step, summary_op])
        summary_writer.add_summary(summary, i)
        if i % save_interval == 0:
            train_acc, train_cm = sess.run([accuracy, conf_matrix])
            saver.save(sess, "conv_rnn/")
            print("At step {}, the accuracy is {}".format(i, train_acc))
            print("The confusion matrix is:")
            print(train_cm)
    saver.save(sess, 'conv_rnn/')
