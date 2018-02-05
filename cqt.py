import sys

# disable display if python(2 or 3)-tkinter not installed
try:
	import tkinter
except:
	import matplotlib
	matplotlib.use('Agg')

import librosa
import librosa.display
import matplotlib.pyplot as plt
import sm
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
    print('{} hs'.format(sr))

    print(y.shape)

    print("Calculating Constant-Q transform") 
    CQT_db = doCQT(y, sr, inDB=True)
    print(np.amax(CQT_db))
    print(np.amin(CQT_db))
    print CQT_db.shape
    print("Plotting")
    librosa.display.specshow(CQT_db, sr=sr,x_axis='time', y_axis='cqt_hz', bins_per_octave=12*BINS_PER_NOTE)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
