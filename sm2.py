#!/usr/bin/env python3

import sys
import os
import time

import librosa
import numpy as np

import cqt
import sight_reading

from intervaltree import IntervalTree, Interval

CQT_AGGREGATE_N = 4

def usage():
	print('Usage: ./sm2.py <MIDI file>')
	print('- looks for a WAV file of the same name (changes the extension)')

'''
- p_wav: path to wav
- returns: (cqt_data, sr, n)
	- cqt_data: [ [ energy ] ] ; 352 bins -> slice index -> energy
	- sr: sample rate
	- n: number of samples
'''
def load_wav(p_wav):
	t0 = time.time()
	print('Loading WAV...')
	(wav_data, sr) = librosa.load(p_wav, sr=None)
	t1 = time.time()
	print('Loaded WAV, data {}, sample rate {} ({:.3f} s).'.format(wav_data.shape, sr, t1 - t0))
	assert(len(wav_data.shape) == 1)
	assert(sr == 44100)
	print('Doing CQT...')
	cqt_data = cqt.doCQT(wav_data, sr, False)
	t2 = time.time()
	print('Done CQT, samples {}, data {} ({:.3f} s).'.format(wav_data.shape[0], cqt_data.shape, t2 - t1))
	assert(cqt_data.shape[0] == 352)
	assert(len(cqt_data.shape) == 2)
	return (cqt_data, sr, wav_data.shape[0])

'''
- p: path to midi
- returns: IntervalTree of IntervalTree(start: float in seconds, end: float in seconds, note number)
'''
def load_midi(p):
	t0 = time.time()
	print('Parsing MIDI...')
	(s1, s2) = sight_reading.process_parse_midi(p)
	assert(len(s1) == 88)
	t1 = time.time()
	print('Loaded MIDI ({:.3f} s).'.format(t1 - t0))
	# not needed
	'''
	print('Processing MIDI data...')
	s3 = sight_reading.process(s1, s2)
	assert(len(s3) == 88)
	t2 = time.time()
	print('Done processing MIDI ({:.3f} s).'.format(t2 - t1))
	'''
	return s2

'''
- heuristic function to check if note overlaps segment
'''
def predicate_note_overlaps(a, b, x, y):
	lower = max(a, x)
	upper = min(b, y)
	return 2 * (upper - lower) / (b - a) <= 1

'''
Note: xs and ys note really named appropriately
'''
def process_iterate(cqt_data, sr, n_samples, midi_it, n):
	# m: number of segments
	m = cqt_data.shape[1] - n + 1
	# b: number of bins
	b = cqt_data.shape[0]
	xs = np.zeros((m, b))
	ys = np.zeros((m, b, n))

	print('Creating target {}, data {}, samples {}, rate {}, n {}...'.format(xs.shape, ys.shape, n_samples, sr, n))

	duration_per_segment = cqt_data.shape[1] * 1.0 / (n_samples * sr)

	t0 = time.time()
	t_midi = 0
	t_cqt = 0
	t_last = t0
	# iterate over segments
	for i in range(m):
		begin = i * duration_per_segment
		end = (i + n) * duration_per_segment
		t_0 = time.time()
		# find midi notes that overlap
		for interval in midi_it.search(begin, end):
			if predicate_note_overlaps(begin, end, interval.begin, interval.end):
				xs[i][interval.data] = 1
		t_1 = time.time()
		# grab the CQT segments
		ys[i] = cqt_data[:, i:i + n]
		t_2 = time.time()
		t_midi += t_1 - t_0
		t_cqt += t_2 - t_1
		if (i + 1) % 100 == 0:
			t_now = time.time()
			print('- processed {} segments ({:.3f}, {:.3f}) ({:.3f}, {:.3f} s)'.format(i + 1, t_midi, t_cqt, t_now - t_last, t_now - t0))
			t_last = t_now
	return (xs, ys)

def process(p):
	assert(p.endswith('.mid'))
	p_base = p[:-4]
	p_wav = p_base + '.wav'

	assert(os.path.isfile(p))
	assert(os.path.isfile(p_wav))

	print('Found {}, {}.'.format(p, p_wav))

	t0 = time.time()
	(cqt_data, sr, n_samples) = load_wav(p_wav)
	t1 = time.time()
	print('WAV took {:.3f} s.'.format(t1 - t0))

	midi_it = load_midi(p)
	#midi_it = IntervalTree()
	t2 = time.time()
	print('MIDI took {:.3f} s.'.format(t2 - t1))

	(xs, ys) = process_iterate(cqt_data, sr, n_samples, midi_it, CQT_AGGREGATE_N)
	t3 = time.time()
	print('Processing took {:.3f} s.'.format(t3 - t2))
	f_data = p_base + '.data.npy'
	f_target = p_base + '.target.npy'
	print('Saving to {}, {}...'.format(f_data, f_target))
	np.save(f_data, ys)
	np.save(f_target, xs)
	t4 = time.time()
	print('Saving took {:.3f} s'.format(t4 - t3))
	print('Done ({:.3f} s).'.format(t4 - t0))

def main(args):
	if len(args) == 0:
		usage()
		return 0
	process(args[0])

if __name__ == '__main__':
	ret = main(sys.argv[1:])
	sys.exit(ret)
