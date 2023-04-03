#!/bin/bash
 
# arg1: benchmark itteration
# arg2: path of stereo image
# arg3: neighbourhood size
# arg4: factor ratio applied to determine the gaussian kernel size

/usr/local/cuda/bin/nvcc denoising.cu -w `pkg-config opencv4 --cflags --libs` cu01_3.cpp -o apps/denoising_cuda
./apps/denoising_cuda $1 $2 $3 $4