import numpy as np
from librosa import core
from sklearn import decomposition
from sklearn import ensemble

class Base_classifier(object):
    def __init__(self, pca_dim = 20, discriminant = ensemble.RandomForestClassifier(), dft_size = 1024, hop_length = 256, window = 'hamming'):
        """
        pca_dim: Number of dimensions in PCA
        discriminant: Classifier used to classify processed data
        """
        self.dft_size = dft_size
        self.hop_length = hop_length
        self.pca_dim = pca_dim
        self.window = window
        self.discriminant = discriminant
        self.pca = None

    def preprocess(self, X, if_train):
        X_stft = np.array([core.stft(x, n_fft = self.dft_size, hop_length = self.hop_length, window = self.window).T for x in X])
        X_stft = np.log(np.abs(X_stft))
        (a, b, c) = X_stft.shape
        X_stft = np.reshape(X_stft, [a*b, -1])
        if if_train:
            self.pca = decomposition.PCA(self.pca_dim).fit(X_stft)
            pca_X = self.pca.transform(X_stft)
        else:
            pca_X = self.pca.transform(X_stft)
        pca_X = np.reshape(pca_X, [a, -1])
        return pca_X

    def fit(self, X, y):
        """
        X: raw audio data, array in shape [n_samples, features]
        y: label, array in shape [n_samples]
        """
        assert len(X) == len(y)
        reduce_X = self.preprocess(X, True)
        self.discriminant.fit(reduce_X, y)

    def predict(self, X):
        """
        X: raw audio data, array in shape [n_samples, features]
        """
        reduce_X = self.preprocess(X, False)
        preds = self.discriminant.predict(reduce_X)
        return preds

def test_baseline():
    fake_X = np.random.random_sample([20, 50000]) + 12
    fake_y = np.hstack((np.zeros(10), np.ones(10)))
    fake_test = np.random.random_sample([10, 50000]) + 10
    baseline = Base_classifier()
    baseline.fit(fake_X, fake_y)
    preds = baseline.predict(fake_test)
    print(preds)

if __name__ == '__main__':
    test_baseline()
