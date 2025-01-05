import librosa
import tensorflow as tf


class DataLoader():
    def __init__(self, data_path, crop_length, num_crops, num_threads, batch_size, normalize=False, channel_count=1):
        self.crop_length = crop_length
        self.fname_queue = tf.train.string_input_producer(tf.train.match_filenames_once(data_path))

        self.num_crops = num_crops
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.channel_count = channel_count
        self.normalize = normalize

    def read_data(self):
        temp = self.fname_queue.dequeue()
        print(temp)
        decoded_audio, _ = librosa.load(temp, sr=None, mono=(self.channel_count == 1))

        if self.normalize:
            decoded_audio = librosa.util.normalize(decoded_audio)

        decoded_audio_cropped = tf.random_crop(decoded_audio, size=[self.crop_length, self.channel_count])

        return decoded_audio_cropped, #this is necessary!!!

    def next_batch(self):
        example_list = [self.read_data() for _ in range(self.num_threads)]
        min_after_dequeue = self.batch_size * self.num_crops
        capacity = min_after_dequeue + 3 * self.batch_size * self.num_crops
        return tf.train.shuffle_batch_join(example_list, self.batch_size, capacity=capacity, enqueue_many=True,
                                           min_after_dequeue=min_after_dequeue, allow_smaller_final_batch=False)