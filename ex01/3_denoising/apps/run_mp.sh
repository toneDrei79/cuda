#!/bin/bash
 
# arg1: benchmark itteration
# arg2: path of stereo image
# arg3: neighbourhood size
# arg4: factor ratio applied to determine the gaussian kernel size

g++ -c -o objs/mp01_3.o -fopenmp `pkg-config opencv4 --cflags --libs` mp01_3.cpp
g++ objs/mp01_3.o -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o apps/denoising_mp
./apps/denoising_mp $1 $2 $3 $4