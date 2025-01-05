# @author Fanbo Xiang
import dotenv
import os
import pandas as pd
import ast
import librosa
import numpy as np
import sys

dotenv.load_dotenv(dotenv.find_dotenv())

# run the command
# cat DATA_DIR=/path/to/data/ > .env
# first to indicator where to seek for data
data_dir = os.environ.get('DATA_DIR')
mfcc_dir = os.environ.get('MFCC_DIR')
mel_dir = os.environ.get('MEL_DIR')

assert data_dir != ''
assert mfcc_dir != ''
assert mel_dir != ''

files = [
    os.path.join(dp, f) for dp, dn, fn in os.walk(data_dir) for f in fn
    if f.endswith('.mp3')
]


# adapted from the FMA dataset
def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

            COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                       ('album', 'date_created'), ('album', 'date_released'),
                       ('artist', 'date_created'), ('artist', 'active_year_begin'),
                       ('artist', 'active_year_end')]

        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
            'category', categories=SUBSETS, ordered=True)

        COLUMNS = [('track', 'license'), ('artist', 'bio'), ('album', 'type'),
                   ('album', 'information')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def get_track_file(idx: int):
    idx = str(idx)
    base = "0" * (6 - len(idx)) + idx
    return os.path.join(data_dir, base[:3], base + '.mp3')


def get_mfcc_file(idx: int):
    idx = str(idx)
    base = "0" * (6 - len(idx)) + idx
    return os.path.join(mfcc_dir, base[:3], base + '.npy')


def get_mel_file(idx: int):
    idx = str(idx)
    base = "0" * (6 - len(idx)) + idx
    return os.path.join(mel_dir, base[:3], base + '.npy')


def get_mfcc_data(idx: int):
    return np.load(get_mfcc_file(idx))


def get_mel_data(idx: int):
    return np.load(get_mel_file(idx))


def get_mfcc_index():
    idx = []
    for f in os.walk(mfcc_dir):
        for f2 in f[2]:
            if f2.endswith('.npy'):
                idx.append(int(f2[:-4]))
    return idx


def load_music_fixed_length(filename: str, fs: int=22050, length: float=30.0):
    music = librosa.load(filename, fs)[0]
    music_length = len(music) / fs
    if (music_length < length / 2):
        sys.stderr.write('Warning, {} is only {}s in duration.\n'.format(filename, music_length))

    true_length = int(fs * length)
    if (len(music) >= true_length):
        return music[:true_length]
    else:
        pad = true_length - len(music)
        padl = int(pad / 2)
        padr = pad - padl
        return np.pad(music, (padl, padr), 'constant')
