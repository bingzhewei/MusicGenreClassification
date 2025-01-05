import argparse
import os
import shutil
import sys

import tensorflow as tf

from SR_like_AutoEncoder import AutoEncoder
from audio_reader import AudioReader
from ops import combine_audio_noise


def save(saver, sess, logdir, step):
    print('Storing checkpoint to {}...'.format(logdir))
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, logdir, global_step=step)
    print('Done.')


def load(saver, sess, logdir):
    print('Restoring checkpoint from {}...'.format(logdir))

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(
            ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print('Global step: {}'.format(global_step))
        print('Restoring...')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Done.')
        return global_step
    else:
        print('No checkpoint found')
        return None


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--saved-model-path', default='./snapshots/', type=str)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--first-run', action='store_true')
    parser.add_argument('--validation-interval', default='50', type=int)
    parser.add_argument('--save-interval', default='10', type=int)
    return parser.parse_args()


def main():
    FLAGS = get_flags()
    print(FLAGS)

    if FLAGS.first_run:
        if os.path.exists(FLAGS.saved_model_path):
            shutil.rmtree(FLAGS.saved_model_path)
        os.makedirs(FLAGS.saved_model_path)

    reader = AudioReader('/media/fx/DATA/598/AnimeOPEDPiano',
                         '/media/fx/DATA/598/AnimeOPED', sample_size=44100*2)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.saved_model_path,
                                               sess.graph)

        reader.start_threads(sess, 8)
        global_step = 0
        audio_batch, noise_batch = reader.dequeue(FLAGS.batch_size)
        pure_batch, noisy_batch = combine_audio_noise(audio_batch, noise_batch, normalize=True)
        pure_batch = tf.expand_dims(pure_batch, 1)
        noisy_batch = tf.expand_dims(noisy_batch, 1)
        auto_encoder = AutoEncoder(noisy_batch, [256, 512], [3, 3], [2, 1], 1, [0.7, 0.7])

        loss = tf.losses.mean_squared_error(pure_batch, auto_encoder)
        tf.summary.scalar('Loss', loss)

        train_op = tf.train.MomentumOptimizer(0.001, 0.8).minimize(loss)

        saver = tf.train.Saver(max_to_keep=20)

        initializer = tf.global_variables_initializer()
        sess.run([initializer])

        if not FLAGS.first_run:
            global_step = load(saver, sess, FLAGS.saved_model_path)
            if global_step is None:
                raise Exception('wtf')

        tf.summary.audio('pure', tf.squeeze(pure_batch, axis=1), 44100)
        tf.summary.audio('noisy', tf.squeeze(noisy_batch, axis=1), 44100)
        tf.summary.audio('restored', tf.squeeze(auto_encoder, axis=1), 44100)

        summary_op = tf.summary.merge_all()

        while True:
            if global_step % FLAGS.save_interval == 0:
                save(saver, sess, FLAGS.saved_model_path, global_step)
            _, summary = sess.run([train_op, summary_op])
            summary_writer.add_summary(summary, global_step=global_step)

            global_step += 1


if __name__ == '__main__':
    main()
