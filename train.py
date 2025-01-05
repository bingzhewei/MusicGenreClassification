import argparse
import datetime
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import time
from audio_reader import AudioReader
from ops import combine_audio_noise
import tensorflow as tf

from discriminator import discriminator
from generator import generator
from model import DataLoader

if __name__ == '__main__':
    starttime_init = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-mask', default='0', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--saved-model-path', default='./snapshots/', type=str)
    parser.add_argument('--data-type-a-path', default=None, type=str)
    parser.add_argument('--data-type-b-path', default=None, type=str)
    parser.add_argument('--num-threads', default=multiprocessing.cpu_count(), type=int)
    parser.add_argument('--num-minibatches', default=100000, type=int)
    parser.add_argument('--save-interval', default=250, type=int)
    parser.add_argument('--validation-interval', default=47, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--gradient-penalty', default=10, type=float)
    parser.add_argument('--loss-coefficient-cycle', default=10, type=float)
    parser.add_argument('--discriminator-train-ratio', default=5, type=int)
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

    reader = AudioReader('/media/fx/DATA/598/AnimeOPEDPiano',
                         '/media/fx/DATA/598/AnimeOPED', sample_size=44100*2)


    with tf.Session() as sess:
        is_training = tf.placeholder(tf.bool, shape=(), name='mode_is_training')
        global_step = tf.placeholder(tf.float32, shape=(), name='global_step')

        audio_batch, noise_batch = reader.dequeue(FLAGS.batch_size)
        pure_batch, noisy_batch = combine_audio_noise(audio_batch, noise_batch, normalize=True)
        DataTypeABatch = pure_batch = tf.expand_dims(pure_batch, 1)
        DataTypeBBatch = noisy_batch = tf.expand_dims(noisy_batch, 1)

        # DataTypeABatch = DataTypeALoader.next_batch()
        # DataTypeBBatch = DataTypeBLoader.next_batch()

        output_A_to_B = generator(DataTypeABatch, reuse=False, name='gen_A_to_B')
        output_B_to_A = generator(DataTypeBBatch, reuse=False, name='gen_B_to_A')

        output_A_to_B_to_A = generator(output_A_to_B, reuse=True, name='gen_B_to_A')
        output_B_to_A_to_B = generator(output_B_to_A, reuse=True, name='gen_A_to_B')

        A_weighted_coeff = tf.random_uniform([FLAGS.batch_size, 1, 1], minval=0, maxval=1)
        B_weighted_coeff = tf.random_uniform([FLAGS.batch_size, 1, 1], minval=0, maxval=1)
        A_hat = A_weighted_coeff * DataTypeABatch + (1.0 - A_weighted_coeff) * output_B_to_A
        B_hat = B_weighted_coeff * DataTypeBBatch + (1.0 - B_weighted_coeff) * output_A_to_B

        discrim_A_to_B_real = discriminator(DataTypeABatch, reuse=False, name='discrim_A')
        discrim_A_to_B_fake = discriminator(output_B_to_A, reuse=True, name='discrim_A')
        discrim_A_hat = discriminator(A_hat, reuse=True, name='discrim_A')

        discrim_A_loss = -(tf.reduce_mean(discrim_A_to_B_real) - tf.reduce_mean(discrim_A_to_B_fake)) + \
                         FLAGS.gradient_penalty * tf.reduce_mean(tf.square(tf.sqrt(
                             tf.reduce_sum(tf.square(tf.gradients(discrim_A_hat, A_hat)[0]),
                                           reduction_indices=[1, 2])) - 1.0))

        discrim_B_to_A_real = discriminator(DataTypeBBatch, reuse=False, name='discrim_B')
        discrim_B_to_A_fake = discriminator(output_A_to_B, reuse=True, name='discrim_B')
        discrim_B_hat = discriminator(B_hat, reuse=True, name='discrim_B')

        discrim_B_loss = -(tf.reduce_mean(discrim_B_to_A_real) - tf.reduce_mean(discrim_B_to_A_fake)) + \
                         FLAGS.gradient_penalty * tf.reduce_mean(tf.square(tf.sqrt(
                             tf.reduce_sum(tf.square(tf.gradients(discrim_B_hat, B_hat)[0]),
                                           reduction_indices=[1, 2])) - 1.0))

        gen_A_loss = -tf.reduce_mean(discrim_A_to_B_fake) + FLAGS.loss_coefficient_cycle * \
                                                            tf.reduce_mean(tf.abs(DataTypeABatch - output_A_to_B_to_A))
        gen_B_loss = -tf.reduce_mean(discrim_B_to_A_fake) + FLAGS.loss_coefficient_cycle *\
                                                            tf.reduce_mean(tf.abs(DataTypeBBatch - output_B_to_A_to_B))

        vars_d = [var for var in tf.trainable_variables() if "discriminator" in var.name]
        vars_g = [var for var in tf.trainable_variables() if "generator" in var.name]

        optimizer_d = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, beta2=0.9).minimize(
            discrim_A_loss + discrim_B_loss, var_list=vars_d)
        optimizer_g = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, beta2=0.9).minimize(
            gen_A_loss + gen_B_loss, var_list=vars_g)

        tf.summary.scalar("loss_d_A_to_B_real", tf.reduce_mean(discrim_A_to_B_real))
        tf.summary.scalar("loss_d_A_to_B_fake", tf.reduce_mean(discrim_A_to_B_fake))
        tf.summary.scalar("d_A_Wasserstein_dist", tf.reduce_mean(discrim_A_to_B_real) - tf.reduce_mean(discrim_A_to_B_fake))
        tf.summary.scalar("loss_d_A", discrim_A_loss)
        tf.summary.scalar("loss_d_A_to_B_identity", tf.reduce_mean(tf.abs(DataTypeABatch - output_A_to_B_to_A)))
        tf.summary.scalar("loss_g_A_to_B", -tf.reduce_mean(discrim_A_to_B_fake))
        tf.summary.scalar("d_A_Gradient_Penalty", tf.reduce_mean(tf.norm(tf.gradients(discrim_A_hat, A_hat), axis=0)- 1.0))

        tf.summary.scalar("loss_d_B_to_A_real", tf.reduce_mean(discrim_B_to_A_real))
        tf.summary.scalar("loss_d_B_to_A_fake", tf.reduce_mean(discrim_B_to_A_fake))
        tf.summary.scalar("d_B_Wasserstein_dist", tf.reduce_mean(discrim_B_to_A_real) - tf.reduce_mean(discrim_B_to_A_fake))
        tf.summary.scalar("loss_d_B", discrim_B_loss)
        tf.summary.scalar("loss_d_B_to_A_identity", tf.reduce_mean(tf.abs(DataTypeBBatch - output_B_to_A_to_B)))
        tf.summary.scalar("loss_g_B_to_A", -tf.reduce_mean(discrim_B_to_A_fake))
        tf.summary.scalar("d_B_Gradient_Penalty", tf.reduce_mean(tf.norm(tf.gradients(discrim_B_hat, B_hat), axis=0) - 1.0))

        tf.summary.scalar("learning_rate", FLAGS.learning_rate)

        tf.summary.audio("output_A_to_B", output_A_to_B, sample_rate=44100, max_outputs=FLAGS.max_tensorboard_audio_outputs)
        tf.summary.audio("output_B_to_A", output_B_to_A, sample_rate=44100, max_outputs=FLAGS.max_tensorboard_audio_outputs)
        tf.summary.audio("output_A_to_B_to_A", output_A_to_B_to_A, sample_rate=44100, max_outputs=FLAGS.max_tensorboard_audio_outputs)
        tf.summary.audio("output_B_to_A_to_B", output_B_to_A_to_B, sample_rate=44100, max_outputs=FLAGS.max_tensorboard_audio_outputs)
        tf.summary.audio("DataTypeABatch", DataTypeABatch, sample_rate=44100, max_outputs=FLAGS.max_tensorboard_audio_outputs)
        tf.summary.audio("DataTypeBBatch", DataTypeBBatch, sample_rate=44100, max_outputs=FLAGS.max_tensorboard_audio_outputs)
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

                for i in range(0, FLAGS.discriminator_train_ratio):
                    _ = sess.run(optimizer_d, feed_dict={is_training: True, global_step: i})

                _, dis_loss_A, dis_loss_B, gen_loss_A, gen_loss_B, summary = sess.run([optimizer_g, discrim_A_loss,
                                                                                      discrim_B_loss, gen_A_loss,
                                                                                      gen_B_loss, merged_summary_op],
                                                      feed_dict={is_training: True, global_step: i})


                print("Iter " + str(i) + ", Disc Loss A: " + "{:.6f}".format(dis_loss_A) + ", Disc Loss B: "
                        + "{:.6f}".format(dis_loss_B) + ", Gen Loss A: " + "{:.6f}".format(gen_loss_A) + ", Gen Loss B: "
                        + "{:.6f}".format(gen_loss_B) + ", Time: " + "{:.3f}".format(time.time() - starttime)
                          + ", Total Time: " + str(datetime.timedelta(seconds=(time.time() - starttime_init))))

                if i % FLAGS.save_inteval == 0:
                    print("Saving!")
                    saver.save(sess, FLAGS.saved_model_path, global_step=i)

                summary_writer.add_summary(summary, global_step=i)

        finally:
            print('Done training for %d epochs, %d total time.' % (i, time.time() - starttime_init))
            coord.request_stop()
            coord.join(threads)
            tboard_proc.terminate()
