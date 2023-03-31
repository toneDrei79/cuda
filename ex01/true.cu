#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "../helper_math.h"

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    int2 coord = {dst_x, dst_y};
    uchar3 left = src(coord.y, coord.x);
    uchar3 right = src(coord.y, coord.x + cols);

    // printf("%d\n", matt);

    dst(dst_y, dst_x).x = uchar(aaa*0.0*left.x + 0.587*left.y + 0.114*left.z);
    dst(dst_y, dst_x).y = 0;
    dst(dst_y, dst_x).z = uchar(0.299*right.x + 0.587*right.y + 0.114*right.z);

    // dst(dst_y, dst_x).x = uchar(matL[0][0]*left.x  + matL[0][1]*left.y  + matL[0][2]*left.z
    //                           + matR[0][0]*right.x + matR[0][1]*right.y + matR[0][2]*right.z);
    // dst(dst_y, dst_x).y = uchar(matL[1][0]*left.x  + matL[1][1]*left.y  + matL[1][2]*left.z
    //                           + matR[1][0]*right.x + matR[1][1]*right.y + matR[1][2]*right.z);
    // dst(dst_y, dst_x).z = uchar(matL[2][0]*left.x  + matL[2][1]*left.y  + matL[2][2]*left.z
    //                           + matR[2][0]*right.x + matR[2][1]*right.y + matR[2][2]*right.z);
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    // float tmp[3] = {1., .5, .2};
    // float tmpt = 0.299;

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols);
}