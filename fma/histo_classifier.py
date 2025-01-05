import numpy as np
from librosa import feature
from collections import Counter
from sklearn import cluster
from sklearn import ensemble


class histo_classifier(object):
    """
    @author Zhihan Xiong
    """
    def __init__(self,
                 pl=44,
                 clusterer=cluster.KMeans(10),
                 classifier=ensemble.RandomForestClassifier()):
        """
        pl: length of each chunk in spectrogram
        clusterer: some sklearn clustering method
        classifier: some sklearn classifier
        """
        self.piece_length = pl
        self.cluster = clusterer
        self.discriminant = classifier

    def cut_songs(self, X):
        (size, fre, time) = X.shape
        num_chunks = int(time / self.piece_length)
        keep_length = self.piece_length * num_chunks
        keep_spec = X[:, :, 0:keep_length]
        split_chunk = np.array(np.split(keep_spec, num_chunks, axis=2))
        cut_result = np.reshape(
            split_chunk, [num_chunks, size, fre * self.piece_length],
            order='F')
        cut_result = np.transpose(cut_result, (1, 0, 2))
        print("Data cut complete.")
        return np.reshape(cut_result, [num_chunks * size, -1]), num_chunks

    def get_histogram(self, X, in_train, mfcc):
        size = len(X)
        if mfcc:
            spec_X = X
        else:
            spec_X = np.array([feature.mfcc(X[i]) for i in range(size)])
        cut_X, num_chunks = self.cut_songs(spec_X)
        if in_train:
            self.cluster.fit(cut_X)
            clus_label = self.cluster.predict(cut_X)
            print("K-means complete.")
        else:
            clus_label = self.cluster.predict(cut_X)
        labels_set = np.unique(clus_label)
        clus_label = np.reshape(clus_label, [size, num_chunks])
        histograms = []
        print("Start creating histograms")
        for i in range(size):
            c = Counter(clus_label[i])
            histograms.append([c[l] for l in labels_set])
        return np.array(histograms, dtype=np.float)

    def train(self, X, y, mfcc = False):
        """
        Assumed shape of X: [size, songs]
        if mfcc = True, then X: [size, frequency, time]
        Assumed shape of y: [size]
        """
        histo_X = self.get_histogram(X, True, mfcc)
        print("Start training classifier")
        self.discriminant.fit(histo_X, y)

    def predict(self, X, mfcc = False):
        """
        Assumed shape of X: [size, songs]
        """
        histo_X = self.get_histogram(X, False, mfcc)
        preds = self.discriminant.predict(histo_X)
        return preds


def test_histo():
    fake_X = np.random.random_sample([100, 88200]) + 5.5
    fake_y = np.hstack((np.array([0 for i in range(50)]),
                        np.array([1 for i in range(50)])))
    fake_test = np.random.random_sample([20, 88200]) + 7.2
    clus = cluster.KMeans(10)
    classi = ensemble.RandomForestClassifier()
    his = histo_classifier(22, clus, classi)
    his.train(fake_X, fake_y)
    preds = his.predict(fake_test)
    print(preds)


if __name__ == '__main__':
    test_histo()
