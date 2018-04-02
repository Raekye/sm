# 3. Import libraries and modules
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(321)  # for reproducibility

from keras import layers
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
from subprocess import call
import librosa
import h5py
import pydot
import time
import sm2
import cqt
import php

LOAD_WEIGHTS = True
TRAIN_WEIGHTS = False
SAVE_WEIGHTS = False
LOAD_LEVEL = "level33"
SAVE_LEVEL = "level34"

class Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.zeros(self.validation_steps)
        targ = np.zeros(self.validation_steps)
        for i in range(self.validation_steps):
            X, y = next(self.validation_data)
            predict[i] = self.model.predict(X)
            targ[i] = y
        predict[predict>=0.5] = 1
        predict[predict<0.5] = 0
        self.precision, self.recall, self.fscore, self.support = precision_recall_fscore_support(targ, predict, average="weighted")
        score = np.mean([1 if np.array_equal(predict[x],targ[x]) else 0 for x in range(predict.shape[0])])
        print("Test accuracy: " + str(score*100) + "%")
        print("Precision: " + str(self.precision))
        print("Recall: " + str(self.recall))
        print("FScore: " + str(self.fscore))
        print("Support: " + str(self.support))
        return
    
def load_song_metadata(paths):
    list_songs = []
    for path in paths:
        for file in os.listdir(path):
            if 'data.preprocessed.npy' in file:
                #print("Loading file " + file)
                name = os.path.join(path,file.replace('data.preprocessed.npy', ''))
                list_songs.append(name)
    
    return list_songs

#deprecated
def load_data_from_folder(path):
    print("Loading data from " + path)
    data = []
    target = []
    for file in os.listdir(path):
        if 'data.preprocessed.npy' in file:
            #print("Loading file " + file)
            newData = np.load(os.path.join(path,file))[::2]
            newTarget = np.load(os.path.join(path,file.replace('data.npy','target.npy')))[::2]
            assert(newTarget.shape[1] == 88)
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

            
def createCNNModel(num_classes=88):
    print("Creating CNN Model")
    model = Sequential()
    
    #model.add(Input(shape=(352,8,1)))
    #model.add(ZeroPadding2D(padding=(0,4)), input_shape=(352,8,1))
    model.add(Conv2D(50, (25,5), input_shape=(352, 7, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(50, (5,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 1)))
    #model.add(Dropout(0.2))
    
    model.add(Conv2D(100, (5,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(100, (5,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 1)))
    
    model.add(Conv2D(100, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(100, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,1)))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='sigmoid'))
    return model

def preprocess(data_path):
    (data, sr, n_samples) = sm2.load_wav(data_path)
    n = sm2.CQT_AGGREGATE_N
    b = data.shape[0]
    m = data.shape[1] - n + 1
    ys = np.zeros((m, b, n))
    for i in range(m):
        ys[i] = np.absolute(data[:,i:i+n])
    data =sm2.preprocess(ys)
    #return (data, onsets)
    return data


def predict(data_path):
    output_name = data_path.replace('.wav', '')
    onsets = None
    if output_name+'.npy' in os.listdir('.'):
        print("Found existing prediction")
        predict = np.load(output_name+'.npy')
        #onsets = np.load(output_name+'.onsets.npy')
    else:
        input_data = preprocess(data_path)
        
        model = createCNNModel(88)
        model.load_weights("weights_" + LOAD_LEVEL)
        predict = model.predict(input_data, verbose=1)
        np.save(output_name, predict)
        #np.save(output_name + '.onsets', onsets)
        
    '''
    top = np.argsort(predict, axis=1)
    for i in range(predict.shape[0]):
        for j in range(top.shape[1]-8):
            predict[i,top[i,j]] = 0
        
    '''
    predict[predict>=0.15] = 1
    predict[predict<0.15] = 0
    
    mid = php.process(predict, sm2.SAMPLE_RATE, cqt.HOP_LENGTH, sm2.CQT_AGGREGATE_N, 3)
    mid.save(output_name + '.mid')
    call(["./synth.sh", output_name+'.mid'])
    plt.imshow(np.transpose(predict)[:,:10000]*255, aspect='auto')
    plt.gca().invert_yaxis()
    #plt.savefig(output_name + '.png')
    #plt.show()
    print("Done")

def train(data_paths):
    #train_size = len(np.load(os.path.join(train_path, 'targets.npy')).item().keys())
    #valid_size = len(np.load(os.path.join(valid_path, 'targets.npy')).item().keys())
    batch_size = 32

    #trainSongs, validSongs, testSongs = load_training_metadata(train_path)
    trainSongs = load_song_metadata([os.path.join(path, 'train') for path in data_paths])
    validSongs = load_song_metadata([os.path.join(path, 'valid') for path in data_paths])
    testSongs = load_song_metadata([os.path.join(path, 'test') for path in data_paths])

    #Define model architecture
    model = createCNNModel(88)
    # Compile model
    epochs = 100  # >>> should be 25+
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    

    metrics = Metrics()
    
    def song_data_generator(list_songs, shuffle=True):
        print("{} songs".format(len(list_songs)))
        total_songs = len(list_songs)
        rnd_idx = np.arange(len(list_songs))
        np.random.shuffle(rnd_idx)
        data = []
        target = []
        total_length = 0
        num_songs = 0
        for idx in rnd_idx:
            #print("loading song {}".format(list_songs[idx]))
            data.append(np.load(list_songs[idx] + 'data.preprocessed.npy'))
            target.append(np.load(list_songs[idx] + 'target.npy'))
            total_length += target[-1].shape[0]
            num_songs +=1
            if total_length > 100000:
                data = np.vstack(data)
                target = np.vstack(target)
                if shuffle:
                    new_idx = np.arange(total_length)
                    np.random.shuffle(new_idx)
                    data = (data)[new_idx]
                    target = (target)[new_idx]
                print("\r{}/{}".format(num_songs,total_songs), end=" ")
                yield data, target, total_length
                data = []
                target = []
                total_length = 0
        if total_length > 0:
            data = np.vstack(data)
            target = np.vstack(target)
            if shuffle:
                new_idx = np.arange(total_length)
                np.random.shuffle(new_idx)
                data = (data)[new_idx]
                target = (target)[new_idx]
            print("\r{}/{} songs".format(num_songs,total_songs), end=" ")
            yield data, target, total_length
        print()
    
    #Fit model on training data
    if LOAD_WEIGHTS:
        model.load_weights("weights_" + LOAD_LEVEL)

    validation_results = np.zeros([epochs, 5])
    train_results = np.zeros([epochs, 2])
    if TRAIN_WEIGHTS:
        best_valid_acc = 0;
        for e in range(epochs):
            print("Epoch {}/{}".format(e+1, epochs))
            print("Training...")
            start = time.time()
            song_acc = []
            song_loss = []
            train_size = 0
            for X_train, Y_train, size in song_data_generator(trainSongs):
                history = model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True, verbose=0)
                song_acc.append(history.history['acc'][0]*X_train.shape[0])
                song_loss.append(history.history['loss'][0]*X_train.shape[0])
                train_size += size
                '''
                model.fit_generator(generator = training_generator, \
                                    steps_per_epoch = len(training_generator), \
                                    validation_data = validation_generator, \
                                    validation_steps = len(validation_generator), \
                                    epochs = epochs, \
                                    verbose = 1, \
                                    workers = 2, \
                                    use_multiprocessing = True, \
                                    max_queue_size = 2000)
                                    #callbacks = [metrics])
                                    '''
            train_end = time.time()
            train_acc = np.array(song_acc)
            train_loss = np.array(song_loss)
            train_acc = np.sum(train_acc)/train_size
            train_loss = np.sum(train_loss)/train_size
            train_results[e] = [train_acc, train_loss]
            print("Train loss: {} - Train acc: {}".format(train_loss,train_acc))
            print("Training time {}s".format(train_end-start))
            print("Validating...")
            predict = []
            targ = []
            for X_val, Y_val, size in song_data_generator(validSongs):
                print("Batch of {}".format(size))
                predict.append(model.predict(X_val))
                targ.append(Y_val)
            predict = np.vstack(predict)
            targ = np.vstack(targ)
            predict[predict>=0.25] = 1
            predict[predict<0.25] = 0
            precision, recall, fscore, support = precision_recall_fscore_support(targ, predict, average="weighted")
            val_acc = np.sum(predict == targ) / predict.size
            if SAVE_WEIGHTS:
                if val_acc > best_valid_acc:
                    best_valid_acc = val_acc
                    model.save_weights("weights_" + SAVE_LEVEL)
            score = np.mean([1 if np.array_equal(predict[x],targ[x]) else 0 for x in range(predict.shape[0])])
            print("Binary accuracy: " + str(val_acc*100) + "%")
            print("Note Precision: " + str(precision))
            print("Note Recall: " + str(recall))
            print("Note FScore: " + str(fscore))
            precision, recall, fscore, support = precision_recall_fscore_support(targ, predict, average="samples")
            print("Frame Accuracy: " + str(score*100) + "%")
            print("Frame Precision: " + str(precision))
            print("Frame Recall: " + str(recall))
            print("Frame FScore: " + str(fscore))
            valid_end = time.time()
            print("Validation time {}s".format(valid_end-train_end))
            validation_results[e] = [val_acc, score, precision, recall, fscore]
        
    if SAVE_WEIGHTS:
        #model.save_weights("weights_" + SAVE_LEVEL)
        plot_model(model, to_file='model_' + SAVE_LEVEL + '.png', show_shapes=True, show_layer_names=True)
        np.save("train_results_" + SAVE_LEVEL + ".npy", train_results)
        np.save("val_results_" + SAVE_LEVEL + ".npy", validation_results)
        
        
    '''
    x_axis = np.arange(epochs)
    train_results = np.load("train_results_level20.npy")
    valid_results = np.load("val_results_level20.npy")
    plt.subplot(211)
    plt.plot(x_axis, train_results[:,0], 'r-')
    plt.plot(x_axis, valid_results[:,0], 'b-')
    plt.subplot(212)
    plt.plot(x_axis, valid_results[:,1], 'g-')
    plt.plot(x_axis, valid_results[:,2], 'r-')
    plt.plot(x_axis, valid_results[:,3], 'b-')
    plt.plot(x_axis, valid_results[:,4], 'm-')
    plt.show()
    
    count = 0
    for x in range(Y_test.shape[0]):
        if not np.any(X_test[x]):
            count += 1
    print("All zero samples: " + str(count))
            '''

    #Evaluate model on test data
    print("Testing...")
    predict = []
    targ = []
    for X_val, Y_val, size in song_data_generator(testSongs, shuffle=False):
        print("Batch of {}".format(size))
        predict.append(model.predict(X_val, verbose=1))
        targ.append(Y_val)
    predict = np.vstack(predict)
    targ = np.vstack(targ)
    
    '''
    for i in np.arange(0.01, 1, 0.01):
        newpred = predict.copy()
        newpred[newpred>=i] = 1.0
        newpred[newpred<i] = 0.0
        precision, recall, fscore, support = precision_recall_fscore_support(targ, newpred, average='weighted')
        val_acc = np.sum(newpred == targ) / newpred.size
        score = np.mean([1 if np.array_equal(newpred[x],targ[x]) else 0 for x in range(newpred.shape[0])])
        print((i.round(2), precision, recall, fscore, val_acc, score))
    '''
    predict[predict>=0.9] = 1.0
    predict[predict<0.9] = 0.0
    best_guess = np.amax([np.sum(predict[x]) if np.array_equal(predict[x],targ[x]) else 0 for x in range(predict.shape[0])])
    print("best guess: " + str(best_guess))
    #np.save("testSongs.level21.predict.npy", predict)
    #np.save("testSongs.level21.truth.npy", targ)
    precision, recall, fscore, support = precision_recall_fscore_support(targ, predict)
    val_acc = np.sum(predict == targ) / predict.size
    score = np.mean([1 if np.array_equal(predict[x],targ[x]) else 0 for x in range(predict.shape[0])])
    print("Binary accuracy: " + str(val_acc*100) + "%")
    print("Frame accuracy: " + str(score*100) + "%")
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("FScore: " + str(fscore))
    print("Support: " + str(support))
    print("Test accuracy: " + str(score*100) + "%")
    
    #php.process(predict, sm2.SAMPLE_RATE, cqt.HOP_LENGTH, sm2.CQT_AGGREGATE_N, 1, 'liz_predict1')
    #php.process(predict, sm2.SAMPLE_RATE, cqt.HOP_LENGTH, sm2.CQT_AGGREGATE_N, 3, 'liz_predict3')
    #php.process(targ, sm2.SAMPLE_RATE, cqt.HOP_LENGTH, sm2.CQT_AGGREGATE_N, 1, 'liz_targ')
    

    
    plt.subplot(211)
    plt.imshow(np.transpose(predict)[:,:10000], aspect='auto')
    plt.gca().invert_yaxis()
    plt.subplot(212)
    plt.imshow(np.transpose(targ)[:,:10000], aspect='auto')
    plt.gca().invert_yaxis()
    #plt.show()

    '''
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
    print("Avg notes: " + str(avg_notes))
    print(miss_type)
    '''

if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2:])
    elif sys.argv[1] == "predict":
        predict(sys.argv[2])
