#!/usr/bin/env python3

import sys
import time
import math
from pprint import pprint

import mido
from intervaltree import IntervalTree, Interval

import piano

WAV_RATE = 44100
WAV_SAMPLING_WINDOW = 882 # WAV_RATE * 20ms

def normalize(x):
	n = x - 21
	if (n < 0) or (88 <= n):
		print('Bad note {}, {}'.format(x, n))
		return None
	return n

def note_name(n):
	return piano.NOTES[n % 12]

'''
- returns: (s1, s2)
	- s1: [ [ (start, end) ] ] ; note number -> array of intervals present
		- start, end: float (seconds)
	- s2: IntervalTree of IntervalTree(start, end, note number)
'''
def process_parse_midi(p):
	kb = [-1] * 88
	sheet = list(map(lambda _: [], range(88)))
	sheet2 = IntervalTree()
	mid = mido.MidiFile(p)
	t = 0
	last_t = 0
	l = []
	i = 0
	print('Opening {}...'.format(p))
	t0 = time.time()
	t1 = t0
	for msg in mid:
		'''
		if msg.type == 'note_on' or msg.type == 'note_off':
			t += msg.time
			if msg.type == 'note_on':
				l.append(note_name(normalize(msg.note)))
			elif msg.type == 'note_off':
				pass
			if msg.time != 0:
				print('{}: {}'.format(t, l))
				l = []
		'''
		t += msg.time
		if msg.type == 'note_on' and msg.velocity != 0:
			n = normalize(msg.note)
			if n is None:
				continue
			#assert(kb[n] == -1)
			if kb[n] == -1:
				kb[n] = t
			else:
				# TODO hmmm
				sheet[n].append((kb[n], t))
				kb[n] = t
		#note_ons with zero velocity also count as valid note_offs
		elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
			n = normalize(msg.note)
			if n is None:
				continue
			if kb[n] < 0:
				# TODO log
				pass
			else:
				if kb[n] == t:
					# TODO log
					print('- note {} had empty interval, skipping'.format(n))
				else:
					sheet[n].append((kb[n], t))
					sheet2.append(Interval(kb[n], t, n))
				kb[n] = -1

		i += 1
		t2 = time.time()
		if i % 50 == 0:
			print('- {} records at {}:{:04.1f} ({:.3f}, {:.3f})...'.format(i, math.floor(t / 60), t % 60, t2 - t1, t2 - t0))
			t1 = t2
		if i == 200:
			#break
			pass
	tn = time.time()
	print('Parsed midi file ({:.3f}).'.format(tn - t0))
	print()
	return (sheet, sheet2)

def process_parse_txt(p):
	kb = [-1] * 88
	sheet = list(map(lambda _: [], range(88)))
	sheet2 = IntervalTree()
	t = 0
	last_t = 0
	l = []
	i = 0
	print('Opening {}...'.format(p))
	t0 = time.time()
	t1 = t0
	with open(p) as noteFile:
		next(noteFile) #skip header line
		for line in noteFile:
			action = line.split()
			assert(len(action)==3)
			begin = float(action[0])
			end = float(action[1])
			n = normalize(int(action[2]))
			sheet[n].append((begin, end))
			sheet2.append(Interval(begin, end, n))
	tn = time.time()
	print('Parsed txt file ({:.3f}).'.format(tn - t0))
	print()
	return (sheet, sheet2)

'''
- returns [ { sample index: [ notes present ] } ] ; note number -> { }
	- note: s2 (return value of process_parse_midi) uses time in seconds,
		s3 (this return value) uses time in samples
'''
def process(s1, s2):
	print('Building data...')
	'''
	print('Printing table...')

	i = 0
	for note in sheet:
		print('{}: {}'.format(i, note))
		i += 1

	print('Done.')
	'''
	t0 = time.time()

	notes_of_interest = range(21, 88)

	# see return value documented above
	# note: [ {} ] * 88 creates an array of 88 references to the same dict
	s3 = [ {} for _ in range(88) ]

	t1 = t0
	for i in notes_of_interest:
		short_windows = 0
		sw2 = 0
		sw3 = 0
		n_intervals = 0
		n_segments = 0
		for (l1, l2) in s1[i]:
			# for each interval, splice it into segments
			# find all the notes present for each segment
			# insert into s3 (structure described above)

			# use floor so don't have off-the-end bugs
			a = math.floor(l1 * WAV_RATE)
			b = math.floor(l2 * WAV_RATE)
			# check for short note
			if (b - a) < WAV_SAMPLING_WINDOW :
				# TODO: log
				short_windows += 1
				continue
			# calculate all the segments of the interval of the note we want to process
			segments = [a]
			if (b - a) < WAV_SAMPLING_WINDOW * 2:
				# quick note, only process "first" and "last" segment
				segments.append(b - WAV_SAMPLING_WINDOW)
			else:
				# longer note, break the interval into 4 segments (can be increased later)
				# TODO: variable number of segments
				d = (b - a) // 4
				for j in range(1, 4):
					segments.append(a + j * d)

			for c in segments:
				# consider the segment [c, c + WAV_RATE)
				# the segment should always contain the original note; will show up in the search
				notes = []
				for i2 in s2.search(c * 1.0 / WAV_RATE, (c + WAV_SAMPLING_WINDOW) * 1.0 / WAV_RATE):
					# check for short segment
					x = math.floor(i2.begin * WAV_RATE)
					y = math.floor(i2.end * WAV_RATE)
					if (y - x) * 2 < WAV_SAMPLING_WINDOW:
						# TODO log
						sw2 += 1
						continue
					# calculate overlap
					lower = max(c, x)
					upper = min(c + WAV_SAMPLING_WINDOW, y)
					# TODO note: currently, the check below covers the previous short segment check
					# but we check for it above separately for logging and possibly different handling in the future
					if (upper - lower) * 2 < WAV_SAMPLING_WINDOW:
						# TODO log
						sw3 += 1
						continue
					# the note overlaps at least half the window; consider it concurrent/polyphonic
					notes.append(i2.data)
					n_segments += 1
				s3[i][c] = notes
			n_intervals += 1
		t2 = time.time()
		print(' - note {} had {} intervals, {} segments, ({}, {}, {}) short segments (took {:.3f} s)'.format(i, n_intervals, n_segments, short_windows, sw2, sw3, t2 - t1))
		t1 = t2
	tn = time.time()
	print('Done building data ({:.3f}).'.format(tn - t0))
	print()
	return s3

def main(args):
	if len(args) == 0:
		print('Halp')
		sys.exit(0)

	(s1, s2) = process_parse_midi(args[0])
	s3 = process(s1, s2)
	print('=== Data')
	for i in range(len(s3)):
		if len(s3[i]) == 0:
			continue
		print('Note {}-{}'.format(i, note_name(i)))
		d = dict(map(lambda x: (x[0], [ '{}-{}'.format(y, note_name(y)) for y in x[1] ]), s3[i].items()))
		pprint(d)
		print()
	print('Done.')

if __name__ == '__main__':
	main(sys.argv[1:])
