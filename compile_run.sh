#!/bin/bash
/usr/local/cuda/bin/nvcc image.cu `pkg-config opencv4 --cflags --libs` imagecpp-linux.cpp -o imagecuda
./imagecuda $1