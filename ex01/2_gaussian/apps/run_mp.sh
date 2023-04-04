#!/bin/bash
 
# arg1: benchmark itteration
# arg2: path of stereo image
# arg3: kernel size
# arg4: sigma

g++ -c -o objs/mp01_2.o -fopenmp `pkg-config opencv4 --cflags --libs` mp01_2.cpp
g++ -c -o objs/gaussain_filtering.o -fopenmp `pkg-config opencv4 --cflags --libs` gaussian_filtering.cpp
g++ objs/mp01_2.o objs/gaussain_filtering.o -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o apps/gaussian_mp
./apps/gaussian_mp $1 $2 $3 $4