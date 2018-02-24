# 3. Import libraries and modules
import sys
import os
import numpy as np
np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import PReLU
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

LOAD_WEIGHTS = False
TRAIN_WEIGHTS = True
SAVE_WEIGHTS = True
LEVEL = "level10"

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
        if 'grand.data.npy' in file:
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
    
    trBatch = int(0.1*data.shape[0])
    rnd_idx = np.arange(trBatch)
    np.random.shuffle(rnd_idx)
    
    trainData = data[0:trBatch]
    trainTarget = target[0:trBatch]
    trainData = trainData[rnd_idx]
    trainTarget = trainTarget[rnd_idx]
    testData = data[trBatch:]
    testTarget = target[trBatch:]
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
    model.add(BatchNormalization())
    model.add(Conv2D(32, (4,2), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (8,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (8,1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='sigmoid'))
    # Compile model
    epochs = 200  # >>> should be 25+
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
        model.load_weights("weights_" + LEVEL)

    if TRAIN_WEIGHTS:
        print("Training...")
        print(X_train.shape)
        print(Y_train.shape)
        model.fit(X_train, Y_train, batch_size=32, validation_data=(X_test, Y_test), epochs=epochs, verbose=2, callbacks=[metrics])
        
    if SAVE_WEIGHTS:
        model.save_weights("weights_" + LEVEL)
        plot_model(model, to_file='model_' + LEVEL + '.png', show_shapes=True, show_layer_names=True)
        
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
    diff = preds - Y_test
    miss_type = {}
    avg_notes = np.count_nonzero(Y_test)/float(Y_test.shape[0])
    for r in diff:
        false_pos = np.count_nonzero(r == 1)
        false_neg = np.count_nonzero(r == -1)
        if false_pos > 0 or false_neg > 0:
            if (false_pos,false_neg) in miss_type:
                miss_type[(false_pos, false_neg)] += 1
            else:
                miss_type[(false_pos, false_neg)] = 1
    miss_type = sorted(miss_type.items(), key=lambda x:x[1], reverse=True)
    print "Test accuracy: " + str(score*100) + "%"
    print "Avg notes: " + str(avg_notes)
    print miss_type



