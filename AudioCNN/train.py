import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
import time

import tensorflow as tf

import AudioCNN.model
from AudioCNN.DataLoader import *

if __name__ == '__main__':
    starttime_init = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-mask', default='1', type=str)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--saved-model-path', default='./snapshots', type=str)
    parser.add_argument('--data-path-train', default='/mnt/data/FMALargeTFRecordsRaw/genre_top/train/*.npz',
                        type=str)
    parser.add_argument('--data-path-val', default='/mnt/data/FMALargeTFRecordsRaw/genre_top/val/*.npz',
                        type=str)
    parser.add_argument('--crop-length', default=22050 * 5, type=int)
    parser.add_argument('--num-threads', default=2, type=int)
    parser.add_argument('--num-minibatches', default=100000, type=int)
    parser.add_argument('--save-interval', default=500, type=int)
    parser.add_argument('--validation-interval', default=503, type=int)
    parser.add_argument('--inital-learning-rate', default=0.001, type=float)
    parser.add_argument('--max-tensorboard-audio-outputs', default=5, type=int)
    parser.add_argument('--first-run', action='store_true')

    FLAGS = parser.parse_args()
    print(FLAGS)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_mask

    FLAGS.saved_model_path = FLAGS.saved_model_path + "_GPU" + FLAGS.gpu_mask + "/"

    if FLAGS.first_run:
        if os.path.exists(FLAGS.saved_model_path):
            shutil.rmtree(FLAGS.saved_model_path)

        os.makedirs(FLAGS.saved_model_path)

    AudioLoaderTrain = DataLoaderNPZ(FLAGS.data_path_train, FLAGS.crop_length, FLAGS.num_threads,
                                     FLAGS.batch_size)
    AudioLoaderVal = DataLoaderNPZ(FLAGS.data_path_val, FLAGS.crop_length, FLAGS.num_threads,
                                   FLAGS.batch_size)

    with tf.Session() as sess:
        label_input = tf.placeholder(tf.int32, shape=[None])
        audio_input = tf.placeholder(tf.float32, shape=[None, FLAGS.crop_length])
        global_step = tf.placeholder(tf.float32, shape=(), name='global_step')

        audio_input_expanded = tf.expand_dims(audio_input, axis=-1)

        if FLAGS.gpu_mask == "0":
            model_output = AudioCNN.model.NetGPU0(audio_input_expanded)
        elif FLAGS.gpu_mask == "1":
            model_output = AudioCNN.model.NetGPU1(audio_input_expanded)
        else:
            model_output = None

        output_op = tf.cast(tf.argmax(model_output, axis=1, name='inference_output'), tf.int32)
        correct_pred = tf.equal(output_op, label_input)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=label_input))

        learning_rate = tf.train.polynomial_decay(FLAGS.inital_learning_rate, global_step,
                                                  FLAGS.num_minibatches, 0.00001,
                                                  power=0.9)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_cross_entropy)

        tf.summary.scalar("Cross Entropy Loss", loss_cross_entropy)
        tf.summary.scalar("Accuracy", accuracy)
        tf.summary.scalar("Learning Rate", learning_rate)
        tf.summary.audio("Input Data", audio_input_expanded, sample_rate=22050,
                         max_outputs=FLAGS.max_tensorboard_audio_outputs)
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=50)
        train_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.saved_model_path, "train"), sess.graph)
        val_summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.saved_model_path, "val"), sess.graph)
        if FLAGS.gpu_mask == "0":
            tboard_proc = subprocess.Popen(shlex.split('tensorboard --port=6006 --logdir=' + FLAGS.saved_model_path))
        elif FLAGS.gpu_mask == "1":
            tboard_proc = subprocess.Popen(shlex.split('tensorboard --port=6007 --logdir=' + FLAGS.saved_model_path))
        else:
            tboard_proc = None

        inital_step_value = 1
        if not FLAGS.first_run:
            print("Resuming from: " + tf.train.latest_checkpoint(FLAGS.saved_model_path))
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.saved_model_path))
            m = re.search(r'\d+$', tf.train.latest_checkpoint(FLAGS.saved_model_path))
            inital_step_value = int(m.group())

        i = inital_step_value

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()),
                 feed_dict={global_step: inital_step_value})

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for i in range(inital_step_value, FLAGS.num_minibatches + 1):
                if coord.should_stop():
                    break

                starttime = time.time()

                if i % FLAGS.validation_interval == 0:
                    val_audio_batch, val_label_batch = AudioLoaderVal.next_batch()
                    val_loss_cross_entropy_result, val_accuracy_result, val_summary_result = sess.run(
                        [loss_cross_entropy, accuracy,
                         merged_summary_op],
                        feed_dict={audio_input: val_audio_batch,
                                   label_input: val_label_batch,
                                   global_step: i})

                    print("Iter " + str(i) + ", Val Loss: " + "{:.6f}".format(val_loss_cross_entropy_result) +
                          ", Val Acc: " + "{:.6f}".format(val_accuracy_result) +
                          ", Time: " + "{:.3f}".format(time.time() - starttime)
                          + ", Total Time: " + str(datetime.timedelta(seconds=(time.time() - starttime_init))))
                    val_summary_writer.add_summary(val_summary_result, global_step=i)

                else:
                    train_audio_batch, train_label_batch = AudioLoaderTrain.next_batch()
                    _, train_loss_cross_entropy_result, train_accuracy_result, train_summary_result = \
                        sess.run([optimizer, loss_cross_entropy, accuracy, merged_summary_op],
                                 feed_dict={audio_input: train_audio_batch,
                                            label_input: train_label_batch,
                                            global_step: i})

                    print("Iter " + str(i) + ", Loss: " + "{:.6f}".format(train_loss_cross_entropy_result) +
                          ", Acc: " + "{:.6f}".format(train_accuracy_result) +
                          ", Time: " + "{:.3f}".format(time.time() - starttime)
                          + ", Total Time: " + str(datetime.timedelta(seconds=(time.time() - starttime_init))))
                    train_summary_writer.add_summary(train_summary_result, global_step=i)

                if i % FLAGS.save_interval == 0:
                    print("Saving!")
                    saver.save(sess, FLAGS.saved_model_path, global_step=i)

        finally:
            print('Done training for %d epochs, %d total time.' % (i - inital_step_value, time.time() - starttime_init))
            coord.request_stop()
            coord.join(threads)
            AudioLoaderTrain.stop()
            AudioLoaderVal.stop()
            tboard_proc.terminate()
