#!/bin/bash
 
# arg1: path of stereo image
# arg2: kernel size

/usr/local/cuda/bin/nvcc gaussian.cu `pkg-config opencv4 --cflags --libs` cu01_2.cpp -o apps/gaussian_cuda
./apps/gaussian_cuda $1 $2