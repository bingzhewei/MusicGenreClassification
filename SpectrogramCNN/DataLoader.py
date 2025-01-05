import librosa
import numpy as np
import tensorflow as tf


class DataLoader():
    def __init__(self, data_path, csv_path, crop_length, num_crops, num_threads, batch_size, channel_count=1):
        self.crop_length = crop_length
        self.csv_path = csv_path
        self.fname_queue = tf.train.string_input_producer(tf.train.match_filenames_once(data_path))
        self.reader = tf.WholeFileReader()

        self.num_crops = num_crops
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.channel_count = channel_count

    def read_data(self):
        fname, value = self.reader.read(self.fname_queue)

        data, sr = librosa.load(fname)
        D = librosa.stft(data)
        D = np.log10(np.absolute(D[0:D.shape[0] / 2 + 1]))

        decoded_audio = tf.contrib.ffmpeg.decode_audio(value, file_format='mp3', samples_per_second=44100,
                                                       channel_count=self.channel_count)

        decoded_audio_cropped = tf.random_crop(decoded_audio, size=[self.crop_length, self.channel_count])

        return fname, decoded_audio_cropped

    def next_batch(self):
        example_list = [self.read_data() for _ in range(self.num_threads)]
        min_after_dequeue = self.batch_size * self.num_crops
        capacity = min_after_dequeue + 3 * self.batch_size * self.num_crops
        return tf.train.shuffle_batch_join(example_list, self.batch_size, capacity=capacity, enqueue_many=True,
                                           min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=True)
