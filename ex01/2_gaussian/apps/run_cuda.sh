#!/bin/bash
 
# arg1: benchmark itteration
# arg2: path of image
# arg3: kernel size
# arg4: sigma

/usr/local/cuda/bin/nvcc gaussian.cu -w `pkg-config opencv4 --cflags --libs` cu01_2.cpp -o apps/gaussian_cuda
# ./apps/gaussian_cuda $1 $2 $3 $4
./apps/gaussian_cuda 100 images/painting.tif 9 2.5