#!/bin/bash
 
# arg1: path of stereo image
# arg2: anaglyph mode (0:true, 1:gray, 2:color, 3:halfcolor, 4:optimized)

g++ -c -o objs/mp01_1.o -fopenmp `pkg-config opencv4 --cflags --libs` mp01_1.cpp
g++ objs/mp01_1.o  objs/anaglyph_mats.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o apps/anaglyph_mp
./apps/anaglyph_mp $1 $2