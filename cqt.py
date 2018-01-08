import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


file_name = sys.argv[1]
print("Loading " + file_name)
y, sr = librosa.load(file_name)

print("Calculating Constant-Q transform")
CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=84*4, bins_per_octave=12*4), ref=np.max)
print("Plotting")
librosa.display.specshow(CQT, x_axis='time', y_axis='cqt_hz', bins_per_octave=48)
plt.colorbar(format='%+2.0f dB')
plt.show()
