#!/usr/bin/env bash

SFS=(grand kawai fazioli-grand giga motif-es6-concert timbres_of_heaven timgm6mb)

if [ -z "$1" ]; then
	echo 'Usage: synth.sh <MIDI file>'
	exit 1
fi
if [ ! -f "$1" ]; then
	echo 'Invalid file.'
	exit 1
fi

BASE="${1%.mid}"

for sf in "${SFS[@]}"
do
	F="$BASE.$sf.wav"
	echo "Creating $F..."
	fluidsynth --disable-lash -F "$F" "$sf.sf2" "$1"
done
