#!/usr/bin/env python3

NOTES = [
	'A',
	'A#',
	'B',
	'C',
	'C#',
	'D',
	'D#',
	'E',
	'F',
	'F#',
	'G',
	'G#'
]

'''
Computes the frequency of the nth note, n in [0, 88).
- middle A is 48
'''
def freq(n):
	return 2 ** ((n - 48) / 12.0) * 440

def main():
	last = freq(0)
	for i in range(88):
		f = freq(i)
		print('Note {} ({}): {:.2f} hz ({:.2f} ms) (diff {:.2f})'.format(i, NOTES[i % 12], f, 1000.0 / f, f - last))
		last = f

if __name__ == '__main__':
	main()
