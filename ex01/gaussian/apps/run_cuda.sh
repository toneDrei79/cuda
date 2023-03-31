#!/bin/bash
 
# arg1: benchmark itteration
# arg2: path of stereo image
# arg3: kernel size
# arg4: sigma

/usr/local/cuda/bin/nvcc gaussian.cu `pkg-config opencv4 --cflags --libs` cu01_2.cpp -o apps/gaussian_cuda
./apps/gaussian_cuda $1 $2 $3 $4