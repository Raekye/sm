import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import sm
import numpy as np

BINS_PER_NOTE = 4

def doCQT(data, sr, inDB=False):
    cqt = librosa.cqt(data, sr=sr, fmin=librosa.note_to_hz('A0'), n_bins=88*BINS_PER_NOTE, bins_per_octave=12*BINS_PER_NOTE)
    if inDB:
        cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
        return cqt_db
    return cqt

if __name__ == "__main__":
    file_name = sys.argv[1]
    print("Loading " + file_name)
    y, sr = librosa.load(file_name, sr=None)
    print "%d hz" % sr

    print y.shape

    print("Calculating Constant-Q transform") 
    CQT_db = doCQT(y, sr)
    print CQT_db.shape
    print("Plotting")
    librosa.display.specshow(CQT_db, sr=sr,x_axis='time', y_axis='cqt_hz', bins_per_octave=12*BINS_PER_NOTE)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
