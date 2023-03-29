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
    uchar3 left = src(coord.y, int(coord.x/2));
    uchar3 right = src(coord.y, int(coord.x/2) + int(cols/2));

    dst(dst_y, dst_x).x = char(0.299*left.x + 0.587*left.y + 0.114*left.z);
    dst(dst_y, dst_x).y = right.y;
    dst(dst_y, dst_x).z = right.z;
}

int divUp(int a, int b) {
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols);
}