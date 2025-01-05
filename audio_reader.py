import os
import threading

import librosa
import numpy as np
import tensorflow as tf


def find_files(directory):
    files = [
        os.path.join(directory, filename) for filename in os.listdir(directory)
    ]
    return [f for f in files if f.endswith('.mp3') or f.endswith('.wav')]


def data_generator(data_files,
                   sample_rate,
                   length,
                   normalize=False,
                   shuffle=True):
    file_index = 0
    while True:
        if (file_index % len(data_files) == 0):
            np.random.shuffle(data_files)

        filename = data_files[file_index % len(data_files)]
        print('File loaded: {}'.format(filename))
        music, _ = librosa.load(filename, sample_rate)
        truncate_length = len(music) - len(music) // length * length
        truncate_head = truncate_length // 2
        truncate_tail = truncate_length - truncate_head
        music = music[truncate_head:-truncate_tail]
        music = music.reshape([-1, 1])

        if normalize:
            music = librosa.util.normalize(music)

        music = music.astype(np.float32)

        chunks = np.split(music, len(music) // length)
        print('File divided into {} chunks'.format(len(chunks)))
        for i, chunk in enumerate(chunks):
            print('{}/{} chunks loaded'.format(i + 1, len(chunks)))
            yield chunk
        file_index += 1


class AudioReader:
    """
    Class for reading all the audio and noise.
    Files given in the directories will be read in a round robin fashion. Each round the order of the files is shuffled.
    Args:
        audio_dir: the directory for all the audio files
        noise_dir: the directory for all the noise files
        sample_rate: audio sample rate
        sample_size: audio length (frames)
        queue_size: size of the random shuffle queue. The queue allows dequeuing when it is at least 90% full
    """

    def __init__(self,
                 audio_dir,
                 noise_dir,
                 sample_rate=44100,
                 sample_size=441000,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.queue_size = queue_size

        self.audio_files = find_files(audio_dir)
        print('Found {} audio files'.format(len(self.audio_files)))
        assert len(self.audio_files) != 0

        self.noise_files = find_files(noise_dir)
        print('Found {} noise files'.format(len(self.noise_files)))
        assert len(self.noise_files) != 0

        self.audio_placeholder = tf.placeholder(tf.float32, (sample_size, 1))
        self.noise_placeholder = tf.placeholder(tf.float32, (sample_size, 1))

        self.queue = tf.RandomShuffleQueue(queue_size, 0.9 * queue_size,
                                           [tf.float32,
                                            tf.float32], [(sample_size, 1),
                                                         (sample_size, 1)])

        self.enqueue = self.queue.enqueue(
            [self.audio_placeholder, self.noise_placeholder])

        self.threads = []

    def dequeue(self, num):
        "dequeues data for feeding into the training process"
        output = self.queue.dequeue_many(num)
        return output

    def thread_main(self, sess):
        "Tensorflow is thread safe so we can run enqueue in a separate thread"
        stop = False
        audio_generator = data_generator(self.audio_files, self.sample_rate, self.sample_size)
        noise_generator = data_generator(self.noise_files, self.sample_rate, self.sample_size)

        while not stop:
            audio = next(audio_generator)
            noise = next(noise_generator)
            sess.run(
                self.enqueue,
                feed_dict={
                    self.audio_placeholder: audio,
                    self.noise_placeholder: noise
                })

    def start_threads(self, sess, n_threads=1):
        "Start the working threads for enqueuing all the stuff."
        # if n_threads != 1:
        #     print(
        #         'Current version only supports 1 thread. Changing thread number to 1...'
        #     )
        # n_threads = 1

        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess, ))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self.threads


# reader = AudioReader('/media/fx/DATA/598/AnimeOPEDPiano', '/media/fx/DATA/598/AnimeOPED')
# with tf.Session() as sess:
#     reader.start_threads(sess)
