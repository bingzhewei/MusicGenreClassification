import glob
import multiprocessing
import random

import numpy as np


class DataLoaderNPZ:
    def __init__(self, data_path, crop_length, num_threads, batch_size, label_key='genre_top'):
        self.crop_length = crop_length
        self.fname_list = glob.glob(data_path)

        self.batch_size = batch_size
        self.num_threads = num_threads
        self.label_key = label_key
        self.secure_random = random.SystemRandom()

        self.queue = multiprocessing.Queue(maxsize=2 * self.batch_size)
        self.proc_list = []

        self.start()

    def start(self):
        self.proc_list = []
        for i in range(0, self.num_threads):
            proc = multiprocessing.Process(target=self.read_data)
            self.proc_list.append(proc)
            proc.start()

    def stop(self):
        for proc in self.proc_list:
            proc.join(0.1)

    def read_data(self):
        while True:
            fname = self.secure_random.choice(self.fname_list)
            with np.load(fname) as npzfile:
                try:
                    crop_index = np.random.randint(0, npzfile['audio'].shape[0] - self.crop_length)
                    self.queue.put(
                        (npzfile['audio'][crop_index:crop_index + self.crop_length], npzfile[self.label_key]))
                except Exception as e:
                    print("Loading Error! Filename: " + fname + " " + str(e))

    def next_batch(self):
        dequeued = [self.queue.get() for _ in range(0, self.batch_size)]
        return np.stack([value[0] for value in dequeued]), np.stack([value[1] for value in dequeued])
