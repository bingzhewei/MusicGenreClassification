import os

import glob
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.transform
import sklearn.linear_model

test = []

lasso = sklearn.linear_model.Lasso(selection='random')

if not os.path.exists("./test"):
    os.mkdir('./test')

for fname in glob.glob("/mnt/data/FMA/fma_small/*/*.mp3")[0:10]:
    print(fname)
    data, sr = librosa.load(fname)
    D = librosa.stft(data)
    D = np.log10(np.absolute(D[0:D.shape[0] / 2 + 1]))
    test.append(D)

out = np.zeros((51, 51), dtype=np.float64)
for x in range(len(test)):
    for y in range(x + 1):
        # if x != y:
        corr = scipy.signal.fftconvolve(test[x], test[y], mode='same')
        out += skimage.transform.resize(corr, (51, 51), mode='reflect')
        print(x, y)
        fig, (ax_orig, ax_template, ax_corr) = plt.subplots(1, 3)
        fig.set_size_inches(9, 3)
        ax_orig.imshow(test[x], cmap='magma')
        ax_orig.set_title('Original')
        ax_orig.set_axis_off()
        ax_template.imshow(test[y], cmap='magma')
        ax_template.set_title('Template')
        ax_template.set_axis_off()
        ax_corr.imshow(corr, cmap='magma')
        ax_corr.set_title('Cross-correlation')
        ax_corr.set_axis_off()
        fig.savefig('./test/' + str(x) + '_' + str(y) + '.png', bbox_inches='tight', dpi=250)
        fig.clf()

fig, (ax_corr) = plt.subplots(1, 1)
fig.set_size_inches(9, 3)
ax_corr.imshow(out, cmap='magma')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()
fig.savefig('./test/' + 'main.png', bbox_inches='tight', dpi=250)
fig.clf()

template = np.zeros((51, 51), dtype=np.float64)
template[25, 25] = 1
row_sums = out.sum()
out = out / row_sums * 100
lasso.fit(template.reshape(1, -1), out.reshape(1, -1))
print(lasso.coef_)
for x in range(len(lasso.coef_)):
    if lasso.coef_[x] != 0:
        print(lasso.coef_[x])
print(lasso.intercept_)

for x in range(len(lasso.intercept_)):
    print(lasso.intercept_[x])
