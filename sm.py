#!/usr/bin/env python3

import sys
import os
import pprint

import numpy as np
import scipy.io.wavfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import piano

WAV_RATE = 44100

'''
- fs: iterable of tuples of (frequency, amplitude)
- ms: duration in milliseconds
'''
def pure_sin(fs, ms):
	x = np.arange(int(WAV_RATE * ms * 1.0 / 1000))
	ws = map(lambda z: z[1] * np.sin((2 * np.pi * z[0] / WAV_RATE) * x), fs)
	return sum(ws)

'''
- d: numpy array of amplitudes
- path: wav file path
'''
def wav_write(d, path):
	scipy.io.wavfile.write(path, WAV_RATE, d.astype(np.int16))

'''
- path: wav file path
'''
def wav_read(path):
	(rate, data) = scipy.io.wavfile.read(path)
	assert(data.dtype == data.dtype.base)
	assert(data.dtype == data.dtype.base.base)
	assert(data.dtype == np.dtype(np.int16))
	assert(rate == WAV_RATE)
	assert(len(data.shape) == 1)
	return data

'''
- d: numpy array
- t: start time in milliseconds
- ms: duration in milliseconds
'''
def data_slice(d, t, ms):
	t0 = int(t * WAV_RATE * 1.0 / 1000)
	t1 = t0 + int(ms * WAV_RATE * 1.0 / 1000)
	if t1 >= d.shape[0]:
		return None
	return d[t0:t1]

'''
Returns iterator of 20ms slices of the data.
- d: numpy array
'''
def data_iter_20ms(d):
	L = 882 # WAV_RATE * 20ms
	for i in range(len(d) - L):
		x = d[i:i + L]
		assert(len(x) == L)
		yield x

'''
- a: amplitude
- ms: duration in milliseconds
- f: directory name
'''
def gen_sad(a, ms, f):
	for i in range(88):
		d = pure_sin([
			(piano.freq(i), a),
		], ms)
		wav_write(d, '{}/pitch-single-{}-{}.wav'.format(f, i, piano.NOTES[i % 12].lower()))

'''
- k: interval
	- major 3: 4
	- perfect 4: 5
	- perfect 5: 7
	- major 6: 9
	- octave: 12
- a1: amplitude 1
- a2: amplitude 2
- ms: duration in milliseconds
- f: directory name
'''
def gen_pitches_interval(k, a1, a2, ms, f):
	for i in range(88 - k):
		d = pure_sin([
			(piano.freq(i), a1),
			(piano.freq(i + k), a2),
		], ms)
		wav_write(d, '{}/pitch-dual-{}-{}.wav'.format(f, i, piano.NOTES[i % 12].lower()))

def help_show():
	print('Hmmm.')
	print('Run with flag `-g` to generate pitches.')
	print('Run with flag `-dw` for whatever.')

def main(args):
	if len(args) == 0:
		help_show()
	elif args[0] == '-g':
		print('Generating WAV files in directory `wavs`...')
		if not os.path.isdir('wavs'):
			os.mkdir('wavs')
		gen_sad(8000, 2000, 'wavs')
		gen_pitches_interval(4, 8000, 3000, 2000, 'wavs')
		print('Done.')
	elif args[0] == '-dw':
		print('Doing whatever.')
		d = pure_sin([(440, 1000)], 1000)
		z = data_slice(d, 0, 1)
		print(z)
		print(data_slice(d, 999, 2))
		d = wav_read('wavs/pitch-single-48-a.wav')
		print(d)
		print(len(d))
		print(len(list(data_iter_20ms(d))))
	else:
		help_show()

if __name__ == '__main__':
	main(sys.argv[1:])
