# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from random import shuffle
import h5py
import pydot
import sm
import piano

LOAD_WEIGHTS = True
SAVE_WEIGHTS = not LOAD_WEIGHTS

def load_data():
	data = []
	test_data = []
	for i in range(24,88-4): # skip first 2 octaves
		f = piano.freq(i)
		f3 = piano.freq(i+4)
		# get a single pitch with amplitude 8000 for 1s
		for amp in range(1000,8001,1000):
			d = sm.pure_sin([(f, amp)], 40)
			y = np.zeros(88)
			y[i] = 1
			for s in sm.data_iter_20ms(d):
				# feed training point s for frequency f
				data.append((s, y))
		# get single pitch + small noise (frequency 200 amplitude 100)
		d = sm.pure_sin([(f, 8000),(f3,4000)], 30)
		y = np.zeros(88)
		y[i] = 0
		y[i+4] = 1
		for s in sm.data_iter_20ms(d):
			# feed training point
			test_data.append((s, y))

	shuffle(data)
	print len(data)
	X_data = np.array([x[0] for x in data])
	Y_data = np.array([x[1] for x in data])
	X_train = X_data[:X_data.shape[0]*8/10]
	X_test = X_data[X_data.shape[0]*8/10:]
	Y_train = Y_data[:Y_data.shape[0]*8/10]
	Y_test = Y_data[Y_data.shape[0]*8/10:]

	X_train = np.array([x[0] for x in data])
	X_test = np.array([x[0] for x in test_data])
	Y_train = np.array([x[1] for x in data])
	Y_test = np.array([x[1] for x in test_data])

	return (X_train, Y_train), (X_test, Y_test)


# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, Y_train), (X_test, Y_test) = load_data()

print X_train.shape
print X_test.shape

# 5. Preprocess input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 8000
X_test /= 8000

# 6. Preprocess class labels
#Y_train = np_utils.to_categorical(y_train, 88)
#Y_test = np_utils.to_categorical(y_test, 88)

print Y_train.shape

# 7. Define model architecture
model = Sequential()

#model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
#model.add(Convolution2D(32, 3, 3, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))

#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(10, activation='softmax'))

model.add(Dense(400, activation='relu', input_dim=882))
model.add(Dense(88, activation='sigmoid'))
model.compile(optimizer='rmsprop',
		loss='binary_crossentropy',
		metrics=['accuracy'])

# 8. Compile model
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

# 9. Fit model on training data
if LOAD_WEIGHTS:
	model.load_weights("weights")
else:
	model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)

# 10. Evaluate model on test data
preds = model.predict(X_test, verbose=1)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
print preds[0]
score = np.mean([1 if np.array_equal(preds[x],Y_test[x]) else 0 for x in range(preds.shape[0])])
print score*100

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
if SAVE_WEIGHTS:
	model.save_weights("weights")

