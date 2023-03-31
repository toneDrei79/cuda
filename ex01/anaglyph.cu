#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"


__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float* mat_l, float* mat_r)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    uchar3 left = src(dst_y, dst_x);
    uchar3 right = src(dst_y, dst_x + cols);

    // GBR order
    dst(dst_y, dst_x).z = char(mat_l[0]*left.z + mat_l[1]*left.y + mat_l[2]*left.x + mat_r[0]*right.z + mat_r[1]*right.y + mat_r[2]*right.x);
    dst(dst_y, dst_x).y = char(mat_l[3]*left.z + mat_l[4]*left.y + mat_l[5]*left.x + mat_r[3]*right.z + mat_r[4]*right.y + mat_r[5]*right.x);
    dst(dst_y, dst_x).x = char(mat_l[6]*left.z + mat_l[7]*left.y + mat_l[8]*left.x + mat_r[6]*right.z + mat_r[7]*right.y + mat_r[8]*right.x);
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float* mat_l, float* mat_r)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    float* dmat_l;
    float* dmat_r;
    size_t mat_size;
    mat_size = sizeof(float) * 9;
    cudaMalloc((void **)&dmat_l, mat_size);
    cudaMalloc((void **)&dmat_r, mat_size);
    cudaMemcpy(dmat_l, mat_l, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dmat_r, mat_r, mat_size, cudaMemcpyHostToDevice);

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, dmat_l, dmat_r);

    cudaFree(dmat_l);
    cudaFree(dmat_r);
}