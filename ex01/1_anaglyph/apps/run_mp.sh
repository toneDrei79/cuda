#!/bin/bash
 
# arg1: benchmark itterations
# arg2: path of stereo image
# arg3: anaglyph mode (0:true, 1:gray, 2:color, 3:halfcolor, 4:optimized)

g++ -c -o objs/mp01_1.o -fopenmp `pkg-config opencv4 --cflags --libs` mp01_1.cpp
g++ -c -o objs/anaglyph_mats.o anaglyph_mats.cpp
g++ objs/mp01_1.o  objs/anaglyph_mats.o  -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o apps/anaglyph_mp
# ./apps/anaglyph_mp $1 $2 $3
./apps/anaglyph_mp 1000 images/stereo03.jpg 2