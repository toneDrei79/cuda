#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernelSize) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    uint3 sum = {0,0,0};
    for (int j=-kernelSize/2; j<kernelSize/2+kernelSize%2; j++) {
        for (int i=-kernelSize/2; i<kernelSize/2+kernelSize%2; i++) {
            char2 coord = {dst_x+i, dst_y+j};
            if (coord.x < 0) coord.x = 0;
            if (coord.y < 0) coord.y = 0;
            if (coord.x >= cols) coord.x = cols-1;
            if (coord.y >= rows) coord.y = rows-1;
            uchar3 val = src(coord.y, coord.x);
            sum.x += val.x;
            sum.y += val.y;
            sum.z += val.z;
        }
    }
    sum.x /= kernelSize * kernelSize;
    sum.y /= kernelSize * kernelSize;
    sum.z /= kernelSize * kernelSize;

    dst(dst_y, dst_x).x = sum.x;
    dst(dst_y, dst_x).y = sum.y;
    dst(dst_y, dst_x).z = sum.z;
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    const int kernelSize = 5;

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, kernelSize);
}