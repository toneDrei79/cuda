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
    const int radius = kernel_size / 2;
    const int size_x = blockDim.x + 2*radius;
    // const int size_y = blockDim.y + 2*radius;
    extern __shared__ uchar3 block_src[]; // the size will be defined when the kernel is run

    const int g_idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int g_idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    const int l_idx_x = threadIdx.x + radius;
    const int l_idx_y = threadIdx.y + radius;

    // copy from src to rectangular buffer on shared memory
    block_src[size_x*l_idx_y + l_idx_x].x = src(g_idx_y, g_idx_x).x;
    block_src[size_x*l_idx_y + l_idx_x].y = src(g_idx_y, g_idx_x).y;
    block_src[size_x*l_idx_y + l_idx_x].z = src(g_idx_y, g_idx_x).z;

    // copy extra pixels to the edge of rectangular buffer in order to perform kernel operations
    if (threadIdx.x < radius && threadIdx.y < radius) { // when near the top-left corner, copy to ...
        // left top
        block_src[size_x*(l_idx_y-radius) + l_idx_x-radius].x = src(g_idx_y-radius, g_idx_x-radius).x;
        block_src[size_x*(l_idx_y-radius) + l_idx_x-radius].y = src(g_idx_y-radius, g_idx_x-radius).y;
        block_src[size_x*(l_idx_y-radius) + l_idx_x-radius].z = src(g_idx_y-radius, g_idx_x-radius).z;
        // right top
        block_src[size_x*(l_idx_y-radius) + l_idx_x+blockDim.x].x = src(g_idx_y-radius, g_idx_x+blockDim.x).x;
        block_src[size_x*(l_idx_y-radius) + l_idx_x+blockDim.x].y = src(g_idx_y-radius, g_idx_x+blockDim.x).y;
        block_src[size_x*(l_idx_y-radius) + l_idx_x+blockDim.x].z = src(g_idx_y-radius, g_idx_x+blockDim.x).z;
        // left buttom
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x-radius].x = src(g_idx_y+blockDim.y, g_idx_x-radius).x;
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x-radius].y = src(g_idx_y+blockDim.y, g_idx_x-radius).y;
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x-radius].z = src(g_idx_y+blockDim.y, g_idx_x-radius).z;
        // right buttom
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x+blockDim.x].x = src(g_idx_y+blockDim.y, g_idx_x+blockDim.x).x;
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x+blockDim.x].y = src(g_idx_y+blockDim.y, g_idx_x+blockDim.x).y;
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x+blockDim.x].z = src(g_idx_y+blockDim.y, g_idx_x+blockDim.x).z;
    }
    if (threadIdx.x < radius) { // when neat the left edge, copy to ...
        // left edge
        block_src[size_x*l_idx_y + l_idx_x-radius].x = src(g_idx_y, g_idx_x-radius).x;
        block_src[size_x*l_idx_y + l_idx_x-radius].y = src(g_idx_y, g_idx_x-radius).y;
        block_src[size_x*l_idx_y + l_idx_x-radius].z = src(g_idx_y, g_idx_x-radius).z;
        // right edge
        block_src[size_x*l_idx_y + l_idx_x+blockDim.x].x = src(g_idx_y, g_idx_x+blockDim.x).x;
        block_src[size_x*l_idx_y + l_idx_x+blockDim.x].y = src(g_idx_y, g_idx_x+blockDim.x).y;
        block_src[size_x*l_idx_y + l_idx_x+blockDim.x].z = src(g_idx_y, g_idx_x+blockDim.x).z;
    }
    if (threadIdx.y < radius) { // when near the top edge, copy to ...
        // top edge
        block_src[size_x*(l_idx_y-radius) + l_idx_x].x = src(g_idx_y-radius, g_idx_x).x;
        block_src[size_x*(l_idx_y-radius) + l_idx_x].y = src(g_idx_y-radius, g_idx_x).y;
        block_src[size_x*(l_idx_y-radius) + l_idx_x].z = src(g_idx_y-radius, g_idx_x).z;
        // buttom edge
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x].x = src(g_idx_y+blockDim.y, g_idx_x).x;
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x].y = src(g_idx_y+blockDim.y, g_idx_x).y;
        block_src[size_x*(l_idx_y+blockDim.y) + l_idx_x].z = src(g_idx_y+blockDim.y, g_idx_x).z;
    }

    __syncthreads();


    float3 rgb_sum = {0, 0, 0};
    float gauss_sum = 0;
    // for each kernel pixel
    for (int j=-radius; j<radius+kernel_size%2; j++)
        for (int i=-radius; i<radius+kernel_size%2; i++)
        {
            // get probability from gaussian equation
            float gauss_val = (1./(2.*PI*pow(sigma, 2.))) * exp(-(pow(i,2.)+pow(j,2.))/(2.*pow(sigma,2.)));
            gauss_sum += gauss_val;

            int2 coord = {l_idx_x+i, l_idx_y+j};
            // if coord is out of image refer the pixel on the edge
            if (g_idx_x+i < 0) coord.x = radius;
            if (g_idx_y+j < 0) coord.y = radius;
            if (g_idx_x+i >= cols) coord.x = threadIdx.x+radius;
            if (g_idx_y+j >= rows) coord.y = threadIdx.y+radius;

            rgb_sum.x += gauss_val * block_src[size_x*coord.y + coord.x].x;
            rgb_sum.y += gauss_val * block_src[size_x*coord.y + coord.x].y;
            rgb_sum.z += gauss_val * block_src[size_x*coord.y + coord.x].z;
        }
    
    dst(g_idx_y, g_idx_x).x = uchar(rgb_sum.x / gauss_sum);
    dst(g_idx_y, g_idx_x).y = uchar(rgb_sum.y / gauss_sum);
    dst(g_idx_y, g_idx_x).z = uchar(rgb_sum.z / gauss_sum);
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, int sigma)
{
    const dim3 block(32, 8); // blockDim.x, blockDim.y
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    printf("%d %d\n", dst.cols, dst.rows);

    // calculate size of shared memory
    const int radius = kernel_size / 2;
    const int size_x = 32 + 2*radius;
    const int size_y = 8 + 2*radius;
    const int size = size_y * size_x * sizeof(char3);

    // dynamicaly define the size of shared memory
    process<<<grid, block, size>>>(src, dst, dst.rows, dst.cols, kernel_size, sigma);
}