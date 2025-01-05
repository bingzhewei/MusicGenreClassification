import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
import time

import tensorflow as tf

from AudioCNN.DataLoader import DataLoader

if __name__ == '__main__':
    starttime_init = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-mask', default='0', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--saved-model-path', default='./snapshots/', type=str)
    parser.add_argument('--data-path', default=None, type=str)
    parser.add_argument('--num-threads', default=3, type=int)
    parser.add_argument('--num-minibatches', default=100000, type=int)
    parser.add_argument('--save-interval', default=250, type=int)
    parser.add_argument('--validation-interval', default=47, type=int)
    parser.add_argument('--inital-learning-rate', default=0.001, type=float)
    parser.add_argument('--max-tensorboard-audio-outputs', default=5, type=int)
    parser.add_argument('--first-run', action='store_true')

    FLAGS = parser.parse_args()
    print(FLAGS)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_mask

    if FLAGS.first_run:
        if os.path.exists(FLAGS.saved_model_path):
            shutil.rmtree(FLAGS.saved_model_path)

        os.makedirs(FLAGS.saved_model_path)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
        global_step = tf.placeholder(tf.float32, shape=(), name='global_step')

        AudioLoader = DataLoader(FLAGS.data_path, 44100 * 10, 5, FLAGS.num_threads, FLAGS.batch_size)
        DataBatch = AudioLoader.next_batch()

        output_op = tf.argmax(output, axis=4, name='inference_output')
        correct_pred = tf.equal(output_op, real_op)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=seg_image_one_hot))

        learning_rate = tf.train.polynomial_decay(FLAGS.inital_learning_rate, global_step,
                                                  FLAGS.num_minibatches, 0.00001,
                                                  power=0.5)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_cross_entropy)

        tf.summary.scalar("Cross Entropy Loss", loss_cross_entropy)
        tf.summary.scalar("Accuracy", loss_cross_entropy)
        tf.summary.scalar("Learning Rate", learning_rate)
        tf.summary.audio("Input Data", DataBatch, sample_rate=44100, max_outputs=FLAGS.max_tensorboard_audio_outputs)
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=50)
        summary_writer = tf.summary.FileWriter(FLAGS.saved_model_path, sess.graph)
        tboard_proc = subprocess.Popen(shlex.split('tensorboard --logdir=' + FLAGS.saved_model_path))

        inital_step_value = 1
        if not FLAGS.first_run:
            print("Resuming from: " + tf.train.latest_checkpoint(FLAGS.saved_model_path))
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.saved_model_path))
            m = re.search(r'\d+$', tf.train.latest_checkpoint(FLAGS.saved_model_path))
            inital_step_value = int(m.group())

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()),
                 feed_dict={is_training: True, global_step: inital_step_value})

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for i in range(inital_step_value, FLAGS.num_minibatches + 1):
                if coord.should_stop():
                    break

                starttime = time.time()

                # if i % FLAGS.validation_interval == 0:
                #     summary = sess.run([merged_summary_op], feed_dict={is_training: False, global_step: i})
                #
                #     print("Iter " + str(i) + ", Val Loss: " + "{:.6f}".format(loss) +
                #           ", Time: " + "{:.3f}".format(time.time() - starttime)
                #           + ", Total Time: " + str(datetime.timedelta(seconds=(time.time() - starttime_init))))

                _, loss_cross_entropy_result, summary_result = sess.run([optimizer, loss_cross_entropy,
                                                                         merged_summary_op],
                                                                        feed_dict={is_training: True, global_step: i})

                print("Iter " + str(i) + ", Loss: " + "{:.6f}".format(loss_cross_entropy_result)
                      + ", Time: " + "{:.3f}".format(time.time() - starttime)
                      + ", Total Time: " + str(datetime.timedelta(seconds=(time.time() - starttime_init))))

                if i % FLAGS.save_inteval == 0:
                    print("Saving!")
                    saver.save(sess, FLAGS.saved_model_path, global_step=i)

                summary_writer.add_summary(summary_result, global_step=i)

        finally:
            print('Done training for %d epochs, %d total time.' % (i - inital_step_value, time.time() - starttime_init))
            coord.request_stop()
            coord.join(threads)
            tboard_proc.terminate()
