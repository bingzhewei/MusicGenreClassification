# @author Fanbo Xiang
import librosa
import numpy as np
from scipy.signal import fftconvolve


def stft(y: np.ndarray, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=np.complex64) -> np.ndarray:
    return librosa.stft(y, n_fft, hop_length, win_length, window, center, dtype)


def mfcc(y: np.ndarray, sr=22050, S=None, n_mfcc=20) -> np.ndarray:
    return librosa.feature.mfcc(y, sr, S, n_mfcc)


def center_of_gravity(spec: np.ndarray, sr=22050, n_fft=2048) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr, n_fft)
    return (freqs @ spec) / np.sum(spec, axis=0)


def spectral_roll_off(spec: np.ndarray, sr=22050, n_fft=2048, thres=0.85) -> np.ndarray:
    freqs = librosa.fft_frequencies(sr, n_fft)
    result = []
    for column in spec.T:
        total = np.sum(column) * thres
        s = 0
        for i, v in enumerate(column):
            s += v
            if s >= total:
                result.append(freqs[i])
                break
        else:
            result.append(freqs[-1])
    return np.array(result)


def spectral_flux(spec: np.ndarray) -> np.ndarray:
    return np.sum((spec[:, 1:] - spec[:, :-1]) ** 2, axis=0)


def auto_corelation(spec: np.ndarray) -> np.ndarray:
    return np.array([fftconvolve(row, row[::-1]) for row in spec])
