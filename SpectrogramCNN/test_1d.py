import os

import glob
import librosa
import numpy as np
import scipy.signal
import skimage.transform

test = []

if not os.path.exists("./test1d"):
    os.mkdir('./test1d')

for fname in glob.glob("/mnt/data/FMA/fma_small/*/*.mp3")[0:1000]:
    print(fname)
    data, sr = librosa.load(fname)
    test.append(np.expand_dims(data, axis=-1))

out = np.zeros((10000, 1), dtype=np.float64)
for x in range(len(test)):
    for y in range(x + 1):
        # if x != y:
        corr = scipy.signal.fftconvolve(test[x], test[y], mode='same')
        out += skimage.transform.resize(corr, (10000, 1), mode='reflect')
        print(x, y)
        # fig, (ax_corr) = plt.subplots(1, 1)
        # fig.set_size_inches(9, 3)
        # ax_orig.stem(test[x], cmap='magma', aspect=20)
        # ax_orig.set_title('Original')
        # ax_orig.set_axis_off()
        # ax_template.stem(test[y], cmap='magma', aspect=20)
        # ax_template.set_title('Template')
        # ax_template.set_axis_off()
# ax_corr.stem(corr, cmap='magma')
#             ax_corr.set_title('Cross-correlation')
#             ax_corr.set_axis_off()
#             fig.savefig('./test1d/' + str(x) + '_' + str(y) + '.png', bbox_inches='tight', dpi=250)
#             fig.clf()
#
# fig, (ax_corr) = plt.subplots(1, 1)
# fig.set_size_inches(9, 3)
# ax_corr.stem(out, cmap='magma', aspect=20)
# ax_corr.set_title('Cross-correlation')
# ax_corr.set_axis_off()
# fig.savefig('./test1d/' + 'main.png', bbox_inches='tight', dpi=250)
# fig.clf()
