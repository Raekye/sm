#!/usr/bin/env bash

find "$@" -maxdepth 1 -name '*.mid' | xargs -L 1 -I % python ./sm2.py % grand disklavier StbgTGd2 AkPnBsdf AkPnBcht AkPnCGdD AkPnStgb SptkBGAm SptkBGCl ENSTDkAm ENSTDkCl
# grand disklavier
# kawai fazioli-grand giga timbres_of_heaven motif-es6-concert timgm6mb
