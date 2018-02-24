#!/usr/bin/env bash

find midi.b -maxdepth 1 -name '*.mid' | xargs -L 1 -I % ./sm2.py % grand kawai fazioli-grand giga timbres_of_heaven motif-es6-concert timgm6mb
