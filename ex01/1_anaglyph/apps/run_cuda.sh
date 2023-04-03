#!/bin/bash
 
# arg1: benchmark itterations
# arg2: path of stereo image
# arg3: anaglyph mode (0:true, 1:gray, 2:color, 3:halfcolor, 4:optimized)

/usr/local/cuda/bin/nvcc anaglyph.cu -w `pkg-config opencv4 --cflags --libs` cu01_1.cpp anaglyph_mats.cpp -o apps/anaglyph_cuda
./apps/anaglyph_cuda $1 $2 $3