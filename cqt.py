import sys

# disable display if python(2 or 3)-tkinter not installed
try:
	import tkinter
except:
	import matplotlib
	matplotlib.use('Agg')

import librosa
import librosa.display
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np

BINS_PER_NOTE = 4
HOP_LENGTH = 512

'''
- data: 1 dim array: [ all samples from WAV file ]
- returns: 352 by ceil((N + 1) / 512) array
	- 4 bins per note
	- (i-th bin, j-th time segment): energy
'''
def doCQT(data, sr, inDB=False):
    cqt = librosa.cqt(data, sr=sr, fmin=librosa.note_to_hz('A0'), n_bins=88*BINS_PER_NOTE, bins_per_octave=12*BINS_PER_NOTE, hop_length=HOP_LENGTH)
    if inDB:
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        return cqt_db
    return np.absolute(cqt)

if __name__ == "__main__":
    file_name = sys.argv[1]
    print("Loading " + file_name)
    y, sr = librosa.load(file_name, sr=None)
    y2, sr2 = librosa.load(file_name, sr=16000)
    print('{} hs'.format(sr))

    print(y.shape)

    print("Calculating Constant-Q transform") 
    CQT_db = doCQT(y, sr, inDB=True)
    CQT_db2 = doCQT(y2, sr2, inDB=True)
    print(np.amax(CQT_db))
    print(np.amin(CQT_db))
    print(CQT_db.shape)
    print("Plotting")
    plt.subplot(211)
    librosa.display.specshow(CQT_db, sr=sr,x_axis='time', y_axis='cqt_hz', bins_per_octave=12*BINS_PER_NOTE)
    plt.subplot(212)
    librosa.display.specshow(CQT_db2, sr=sr2,x_axis='time', y_axis='cqt_hz', bins_per_octave=12*BINS_PER_NOTE)
    #plt.show()
