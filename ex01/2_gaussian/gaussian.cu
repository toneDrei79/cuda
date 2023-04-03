#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"


#define PI 3.1415

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int kernel_size, int sigma)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    float3 rgb_sum = {0, 0, 0};
    float gauss_sum = 0;
    // for each kernel pixel
    for (int j=-kernel_size/2; j<kernel_size/2+kernel_size%2; j++)
        for (int i=-kernel_size/2; i<kernel_size/2+kernel_size%2; i++)
        {
            // get probability from gaussian equation
            float gauss_val = (1./(2.*PI*pow(sigma, 2.))) * exp(-(pow(i,2.)+pow(j,2.))/(2.*pow(sigma,2.)));
            gauss_sum += gauss_val;

            int2 coord = {dst_x+i, dst_y+j};
            // if coord is out of image refer the pixel on the edge
            if (coord.x < 0)  coord.x = 0;
            if (coord.y < 0) coord.y = 0;
            if (coord.x >= cols) coord.x = cols - 1;
            if (coord.y >= rows) coord.y = rows - 1;

            rgb_sum.x += gauss_val * src(coord.y, coord.x).x;
            rgb_sum.y += gauss_val * src(coord.y, coord.x).y;
            rgb_sum.z += gauss_val * src(coord.y, coord.x).z;
        }
    
    dst(dst_y, dst_x).x = uchar(rgb_sum.x / gauss_sum);
    dst(dst_y, dst_x).y = uchar(rgb_sum.y / gauss_sum);
    dst(dst_y, dst_x).z = uchar(rgb_sum.z / gauss_sum);
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, int sigma)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, kernel_size, sigma);
}