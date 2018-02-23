# 3. Import libraries and modules
import sys
import os
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils, plot_model, normalize
from keras.datasets import mnist
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from random import shuffle
from numpy import fft
import librosa
import h5py
import pydot
import sm
import piano

LOAD_WEIGHTS = False
SAVE_WEIGHTS = not LOAD_WEIGHTS

class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        predict[predict>=0.5] = 1
        predict[predict<0.5] = 0
        targ = self.validation_data[1]
        self.precision, self.recall, self.fscore, self.support = precision_recall_fscore_support(targ, predict, average="weighted")
        score = np.mean([1 if np.array_equal(predict[x],targ[x]) else 0 for x in range(predict.shape[0])])
        print "Test accuracy: " + str(score*100) + "%"
        print("Precision: " + str(self.precision))
        print("Recall: " + str(self.recall))
        print("FScore: " + str(self.fscore))
        print("Support: " + str(self.support))
        return

def load_data_from_folder(path):
    print("Loading data from " + path)
    data = []
    target = []
    for file in os.listdir(path):
        if 'data.npy' in file:
            newData = np.load(os.path.join(path,file))
            newTarget = np.load(os.path.join(path,file.replace('data.npy','target.npy')))[:,0:88]
            #data = np.vstack((data, newData))
            #target = np.vstack((target, newTarget))
            data.append(newData)
            target.append(newTarget)
    data = np.vstack(data)
    target = np.vstack(target)
    print(data.shape)
    print(target.shape)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    
    trBatch = int(0.8*len(rnd_idx))
    
    trainData = data[rnd_idx[0:trBatch]]
    testData = data[rnd_idx[trBatch:]]
    trainTarget = target[rnd_idx[0:trBatch]]
    testTarget = target[rnd_idx[trBatch:]]
    print(trainData.shape)
    print(testData.shape)
    
    return trainData, trainTarget, testData, testTarget

            
def preprocess(data):
    print("Preprocessing")
    data = librosa.amplitude_to_db(data, ref=np.max)
    data = data - np.amin(data)
    data = data / np.amax(data)
    print(np.amax(data))
    print(np.amin(data))
    print(data.shape)
    return np.expand_dims(data,axis=-1)

def createCNNModel(num_classes=88):
    print("Creating CNN Model")
    model = Sequential()
    
    model.add(Conv2D(32, (4,2), input_shape=(352, 4, 1), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(32, (4,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    model.add(Conv2D(64, (8,1), activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(64, (8,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))
    # Compile model
    epochs = 3  # >>> should be 25+
    lrate = 0.01
    decay = lrate/epochs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'binary_accuracy'])
    print(model.summary())
    return model, epochs

if __name__ == "__main__":
    load_path = sys.argv[1]

    #Load data into train and test sets
    X_train, Y_train, X_test, Y_test = load_data_from_folder(load_path)

    #Preprocess input data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    #normalize data if needed
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    #Define model architecture
    model, epochs = createCNNModel(88)

    metrics = Metrics()
    
    #Fit model on training data
    if LOAD_WEIGHTS:
        model.load_weights("weights_level2")
    else:
        print("Training...")
        print(X_train.shape)
        print(Y_train.shape)
        model.fit(X_train, Y_train, batch_size=32, validation_split=0.20, epochs=epochs, verbose=1, callbacks=[metrics])
        
    if SAVE_WEIGHTS:
        model.save_weights("weights_level2")
        
    count = 0
    for x in range(Y_test.shape[0]):
        if not np.any(X_test[x]):
            count += 1
            
    print("All zero samples: " + str(count))

    #Evaluate model on test data
    print("Testing...")
    preds = model.predict(X_test, verbose=1)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    precision, recall, fscore, support = precision_recall_fscore_support(Y_test, preds)
    print(precision)
    print(recall)
    print(fscore)
    print(support)
    score = np.mean([1 if np.array_equal(preds[x],Y_test[x]) else 0 for x in range(preds.shape[0])])
    print "Test accuracy: " + str(score*100) + "%"

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


