#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float* matL, float* matR) {
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    uchar3 left = src(dst_y, dst_x);
    uchar3 right = src(dst_y, dst_x + cols);

    // GBR order
    dst(dst_y, dst_x).z = char(matL[0]*left.z + matL[1]*left.y + matL[2]*left.x + matR[0]*right.z + matR[1]*right.y + matR[2]*right.x);
    dst(dst_y, dst_x).y = char(matL[3]*left.z + matL[4]*left.y + matL[5]*left.x + matR[3]*right.z + matR[4]*right.y + matR[5]*right.x);
    dst(dst_y, dst_x).x = char(matL[6]*left.z + matL[7]*left.y + matL[8]*left.x + matR[6]*right.z + matR[7]*right.y + matR[8]*right.x);
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float* matL, float* matR) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    float* dMatL;
    float* dMatR;
    size_t matSize;
    matSize = sizeof(float) * 9;
    cudaMalloc((void **)&dMatL, matSize);
    cudaMemcpy(dMatL, matL, matSize, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dMatR, matSize);
    cudaMemcpy(dMatR, matR, matSize, cudaMemcpyHostToDevice);

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, dMatL, dMatR);
}