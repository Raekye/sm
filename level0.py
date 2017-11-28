# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from random import shuffle
import pydot
import sm
import piano

def load_data():
	data = []
	test_data = []
	for i in range(88): # skip first 2 octaves
		f = piano.freq(i)
		# get a single pitch with amplitude 8000 for 1s
		d = sm.pure_sin([(f, 4000)], 50)
		for s in sm.data_iter_20ms(d):
			# feed training point s for frequency f
			data.append((s, i))
		# get single pitch + small noise (frequency 200 amplitude 100)
		d = sm.pure_sin([(f, 4000), (200, 500)], 30)
		for s in sm.data_iter_20ms(d):
			# feed training point
			test_data.append((s, i))
	
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
(X_train, y_train), (X_test, y_test) = load_data()

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape
 
# 5. Preprocess input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 8000
X_test /= 8000
 
# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 88)
Y_test = np_utils.to_categorical(y_test, 88)

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
model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)
 
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=1)
print score

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

