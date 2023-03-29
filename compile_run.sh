#!/bin/bash
/usr/local/cuda/bin/nvcc $1 `pkg-config opencv4 --cflags --libs` imagecpp-linux.cpp -o imagecuda
./imagecuda $2