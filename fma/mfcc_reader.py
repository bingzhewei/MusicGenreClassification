import utils
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import threading


def train_generator(train_indices, train_y, feature_func):
    classes = len(np.unique(train_y))
    while True:
        print('new epoch')
        idx = np.arange(len(train_indices))
        np.random.shuffle(idx)
        train_indices = train_indices[idx]
        train_y = train_y[idx]
        for index, y in zip(train_indices, train_y):
            X = feature_func(index)
            one_hot_y = np.zeros(classes)
            one_hot_y[y] = 1
            yield X, one_hot_y


class Loader():
    def __init__(self, queue_size=50, feature='mfcc'):
        self.feature = feature
        self.init_loader()
        self.train_placeholder = tf.placeholder(tf.float32, (self.width,
                                                             self.length))
        self.train_label_placeholder = tf.placeholder(tf.float32,
                                                      (self.n_classes, ))
        self.queue = tf.RandomShuffleQueue(queue_size, 0.9 * queue_size, [
            tf.float32, tf.float32
        ], [(self.width, self.length), (self.n_classes, )])

        self.enqueue = self.queue.enqueue(
            [self.train_placeholder, self.train_label_placeholder])
        # test queue stuff

    def init_loader(self):
        training_tracks = utils.load('./training_tracks.csv')
        testing_tracks = utils.load('./testing_tracks.csv')

        training_labels = training_tracks[('track', 'genre_top')]
        testing_labels = testing_tracks[('track', 'genre_top')]

        if self.feature == 'mfcc':
            self.feature_function = utils.get_mfcc_data
        elif self.feature == 'mel':
            self.feature_function = utils.get_mel_data

        self.train_indices = training_tracks.index
        self.test_indices = testing_tracks.index

        # self.train_X = np.array(
        #     [feature_function(idx) for idx in training_tracks.index])
        # self.test_X = np.array(
        #     [feature_function(idx) for idx in testing_tracks.index])

        le = LabelEncoder().fit(training_labels)
        self.train_y = le.transform(training_labels)
        self.test_y = le.transform(testing_labels)

        print('train size:', len(self.train_y), 'test size:', len(self.test_y))

        X0 = self.feature_function(self.train_indices[0])

        self.n_classes = len(le.classes_)
        self.width, self.length = X0.shape

    def dequeue(self, num):
        return self.queue.dequeue_many(num)

    def thread_train(self, sess, test=False):
        stop = False
        if not test:
            train_gen = train_generator(self.train_indices, self.train_y, self.feature_function)
        else:
            train_gen = train_generator(self.test_indices, self.test_y, self.feature_function)

        while not stop:
            X, y = next(train_gen)
            sess.run(
                self.enqueue,
                feed_dict={
                    self.train_placeholder: X,
                    self.train_label_placeholder: y
                })

    def start_threads(self, sess, test=False):
        thread_train = threading.Thread(
            target=self.thread_train, args=(sess, test))
        thread_train.daemon = True
        thread_train.start()
