# 3. Import libraries and modules
import sys
import os
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils, plot_model, normalize
from keras.datasets import mnist
from sklearn import preprocessing
from random import shuffle
from numpy import fft
import h5py
import pydot
import sm
import piano

LOAD_WEIGHTS = True
SAVE_WEIGHTS = not LOAD_WEIGHTS

def load_data_from_folder(dir):
	data = []
	for file in os.listdir(dir):
		synth = file.split('-')[1]
		if synth == 'synth_samp':
			note = int(file.split('-')[2]) # 0-88
			print os.path.join(dir,file)
			wav = sm.wav_read(os.path.join(dir,file))
			wav = sm.data_slice(wav, 0, 100)
			f = piano.freq(note)
			y = np.zeros(88)
			y[note] = 1
			for s in sm.data_iter_20ms(wav):
				data.append((s, y))
	shuffle(data)

	X_data = np.array([x[0] for x in data])
	Y_data = np.array([x[1] for x in data])
	X_train = X_data[:X_data.shape[0]*8/10]
	X_test = X_data[X_data.shape[0]*8/10:]
	Y_train = Y_data[:Y_data.shape[0]*8/10]
	Y_test = Y_data[Y_data.shape[0]*8/10:]
	return (X_train, Y_train), (X_test, Y_test)




def load_data():
	data = []
	test_data = []
	for i in range(24,88-4): # skip first 2 octaves
		f = piano.freq(i)
		f3 = piano.freq(i+4)
		# get a single pitch with amplitude 8000 for 1s
		for amp in range(4000,8001,1000):
			d = sm.pure_sin([(f, amp)], 40)
			y = np.zeros(88)
			y[i] = 1
			for s in sm.data_iter_20ms(d):
				# feed training point s for frequency f
				data.append((s, y))
		# get single pitch + small noise (frequency 200 amplitude 100)
		d = sm.pure_sin([(f, 4000),(f3,4000)], 30)
		y = np.zeros(88)
		y[i] = 1
		y[i+4] = 1
		for s in sm.data_iter_20ms(d):
			# feed training point
			data.append((s, y))

	shuffle(data)
	print len(data)
	X_data = np.array([x[0] for x in data])
	Y_data = np.array([x[1] for x in data])
	X_train = X_data[:X_data.shape[0]*8/10]
	X_test = X_data[X_data.shape[0]*8/10:]
	Y_train = Y_data[:Y_data.shape[0]*8/10]
	Y_test = Y_data[Y_data.shape[0]*8/10:]

	#X_train = np.array([x[0] for x in data])
	#X_test = np.array([x[0] for x in test_data])
	#Y_train = np.array([x[1] for x in data])
	#Y_test = np.array([x[1] for x in test_data])

	return (X_train, Y_train), (X_test, Y_test)


if __name__ == "__main__":
    #Load data into train and test sets
    (X_train, Y_train), (X_test, Y_test) = load_data_from_folder('wavs2/')

    #Preprocess input data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = np.array([abs(fft.fft(X_train[i])/882)for i in range(X_train.shape[0])])
    X_test = np.array([abs(fft.fft(X_test[i])/882) for i in range(X_test.shape[0])])

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Define model architecture
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=882))
    model.add(Dense(88, activation='sigmoid'))

    #Compile model
    model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

    #Fit model on training data
    if LOAD_WEIGHTS:
        model.load_weights("weights")
    else:
        print("Training...")
        model.fit(X_train, Y_train, batch_size=32, epochs=2, verbose=1)

    #Evaluate model on test data
    print("Testing...")
    preds = model.predict(X_test, verbose=1)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    score = np.mean([1 if np.array_equal(preds[x],Y_test[x]) else 0 for x in range(preds.shape[0])])
    print "Test accuracy: " + str(score*100) + "%"

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    if SAVE_WEIGHTS:
        model.save_weights("weights_fft2")

