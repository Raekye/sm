#!/usr/bin/env python3

import sys
from functools import reduce
from collections import deque

import numpy as np
import mido
import librosa
from intervaltree import IntervalTree

import sm2
import cqt

TICKS_PER_BEAT = 480

def midi_create():
	mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
	track = mido.MidiTrack()
	mid.tracks.append(track)

	track.append(mido.Message('program_change', program=12, time=0))

	return (mid, track)

def midi_note(track, on, note, duration):
	d = mido.second2tick(duration, TICKS_PER_BEAT, mido.bpm2tempo(60))
	track.append(mido.Message('note_on' if on else 'note_off', note=note + 21, time=int(d), velocity=64))

'''
Predicate to consider note played.
- return: 88-vector of 0 or 1
'''
def gandalf(history):
	tachi = reduce(lambda acc, x: acc + x, history, np.zeros(88)) / len(history)
	fn = np.vectorize(lambda x: 0 if x < 0.5 else 1)
	return fn(tachi)

'''
- data: [ [ <1 or 0> ] ] ; segment_index -> [ 88-vector of note on/off ]
- agg_length1: number of frames per segment in preprocessing
- agg_length2: last-n segments to average
'''
def process(data, freq, hop_length, agg_length1, agg_length2):
	assert(len(data) > 0)
	# see sm2.process_iterate
	n_frames = len(data) + agg_length1 - 1
	frame_duration = librosa.frames_to_time(1, freq, hop_length)[0]

	(mid, track) = midi_create()

	# initialize last-n history with segment 0
	# careful to do deep copy
	history = deque([ np.array(data[0]) for _ in range(agg_length2 - 1) ])
	kb = [-1] * 88
	last_msg = 0

	for (i, segment) in enumerate(data):
		history.append(np.array(segment))
		# add last-n states together
		not_ninjas = gandalf(history)
		for note in range(88):
			if not_ninjas[note] == 0:
				if kb[note] == -1:
					# note not being played
					pass
				else:
					midi_note(track, False, note, (i - last_msg) * frame_duration)
					last_msg = i
					kb[note] = -1
			elif not_ninjas[note] == 1:
				if kb[note] == -1:
					midi_note(track, True, note, (i - last_msg) * frame_duration)
					last_msg = i
					kb[note] = i
				else:
					# note still being played
					pass
			else:
				assert(False)
		history.popleft()
	for note in range(88):
		# could pull this out of the loop, but I like keeping it visually namespaced inside the loop
		# (note: python doesn't have block level scoping, so this is accessible outside the loop still)
		n = len(data)
		if kb[note] != -1:
			midi_note(track, False, note, (n - last_msg) * frame_duration)
			last_msg = n

	mid.save('foobar.mid')

def main(args):
	process([ ([0] * 48) + [1] + ([0] * 39) ] * 3000, sm2.SAMPLE_RATE, cqt.HOP_LENGTH, sm2.CQT_AGGREGATE_N, sm2.CQT_AGGREGATE_N)

if __name__ == '__main__':
	main(sys.argv[1:])
