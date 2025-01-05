import argparse
import csv
import glob
import itertools
import multiprocessing
import os

import librosa
import numpy as np


def convert_mp3_to_npz_genre_top(fname, out_path, lookup_table):
    fnameout = os.path.join(out_path, os.path.splitext(fname)[0].split("/")[-1] + ".npz")
    mp3_id = int(os.path.splitext(fname)[0].split("/")[-1])

    if mp3_id not in lookup_table:
        return

    mp3file_decoded, _ = librosa.core.load(fname, sr=None)
    np.savez_compressed(fnameout, genre_top=np.array(lookup_table[mp3_id], dtype=np.int32), audio=mp3file_decoded)


def convert_mp3_to_npz_genre_all(fname, out_path, lookup_table):
    fnameout = os.path.join(out_path, os.path.splitext(fname)[0].split("/")[-1] + ".npz")
    mp3_id = int(os.path.splitext(fname)[0].split("/")[-1])

    if mp3_id not in lookup_table:
        return

    mp3file_decoded, _ = librosa.core.load(fname, sr=None)
    label = np.array(lookup_table[mp3_id], dtype=np.int32)
    label_one_hot = np.zeros(len(genre_id_to_training_label), dtype=np.int32)
    label_one_hot[label] = 1
    np.savez_compressed(fnameout, genre_all=label_one_hot, audio=mp3file_decoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('srcdir', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('csvdir', type=str)
    parser.add_argument('genredir', type=str)

    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.outdir):
        os.makedirs(FLAGS.outdir)
    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_top")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_top"))
    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_all")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_all"))

    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_top/train")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_top/train"))
    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_all/train")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_all/train"))
    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_top/val")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_top/val"))
    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_all/val")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_all/val"))
    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_top/test")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_top/test"))
    if not os.path.exists(os.path.join(FLAGS.outdir, "genre_all/test")):
        os.makedirs(os.path.join(FLAGS.outdir, "genre_all/test"))

    fname_all = glob.glob(os.path.join(FLAGS.srcdir, "**/*.mp3"), recursive=True)

    genre_label_to_id = {}
    genre_id_to_training_label = {}
    with open(FLAGS.genredir, "r") as genrefile:
        csvreader = csv.reader(genrefile)
        next(csvreader)  # skip the header

        for line in csvreader:
            genre_label_to_id[line[3]] = int(line[0])
            genre_id_to_training_label[int(line[0])] = int(line[5])

    genre_top_train = {}
    genre_all_train = {}
    genre_top_val = {}
    genre_all_val = {}
    genre_top_test = {}
    genre_all_test = {}
    with open(FLAGS.csvdir, "r") as f:
        csvreader = csv.reader(f)
        next(csvreader)
        next(csvreader)
        next(csvreader)  # skip the header

        for line in csvreader:
            if line[31] == "training":  # Col AF
                if line[40] != '':
                    genre_top_train[int(line[0])] = genre_id_to_training_label[genre_label_to_id[line[40]]]
                if line[42][1:-1].split(",") != [""]:
                    genre_all_train[int(line[0])] = [genre_id_to_training_label[int(x)] for x in
                                                     line[42][1:-1].split(",")]
            elif line[31] == "validation":
                if line[40] != '':
                    genre_top_val[int(line[0])] = genre_id_to_training_label[genre_label_to_id[line[40]]]
                if line[42][1:-1].split(",") != [""]:
                    genre_all_val[int(line[0])] = [genre_id_to_training_label[int(x)] for x in
                                                   line[42][1:-1].split(",")]
            elif line[31] == "test":
                if line[40] != '':
                    genre_top_test[int(line[0])] = genre_id_to_training_label[genre_label_to_id[line[40]]]
                if line[42][1:-1].split(",") != [""]:
                    genre_all_test[int(line[0])] = [genre_id_to_training_label[int(x)] for x in
                                                    line[42][1:-1].split(",")]
            else:
                print('fail!')

    with multiprocessing.Pool() as p:
        p.starmap(convert_mp3_to_npz_genre_top,
                  zip(fname_all, itertools.repeat(os.path.join(FLAGS.outdir, "genre_top/train"), len(fname_all)),
                      itertools.repeat(genre_top_train, len(fname_all))))
        p.starmap(convert_mp3_to_npz_genre_all,
                  zip(fname_all, itertools.repeat(os.path.join(FLAGS.outdir, "genre_all/train"), len(fname_all)),
                      itertools.repeat(genre_all_train, len(fname_all))))
        p.starmap(convert_mp3_to_npz_genre_top,
                  zip(fname_all, itertools.repeat(os.path.join(FLAGS.outdir, "genre_top/val"), len(fname_all)),
                      itertools.repeat(genre_top_val, len(fname_all))))
        p.starmap(convert_mp3_to_npz_genre_all,
                  zip(fname_all, itertools.repeat(os.path.join(FLAGS.outdir, "genre_all/val"), len(fname_all)),
                      itertools.repeat(genre_all_val, len(fname_all))))
        p.starmap(convert_mp3_to_npz_genre_top,
                  zip(fname_all, itertools.repeat(os.path.join(FLAGS.outdir, "genre_top/test"), len(fname_all)),
                      itertools.repeat(genre_top_test, len(fname_all))))
        p.starmap(convert_mp3_to_npz_genre_all,
                  zip(fname_all, itertools.repeat(os.path.join(FLAGS.outdir, "genre_all/test"), len(fname_all)),
                      itertools.repeat(genre_all_test, len(fname_all))))
