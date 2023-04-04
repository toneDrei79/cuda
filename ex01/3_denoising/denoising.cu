#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

#include "helper_math.h"


#define MAX_KERNEL 15
#define SIGMA 1.5
#define PI 3.1415

__device__ void gaussian_filtering(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst,
                        int rows, int cols,
                        int kernel_size, int sigma,
                        int x, int y)
{
    float3 rgb_sum = {0, 0, 0};
    float gauss_sum = 0;
    // for each kernel pixel
    for (int j=-kernel_size/2; j<kernel_size/2+kernel_size%2; j++)
        for (int i=-kernel_size/2; i<kernel_size/2+kernel_size%2; i++)
        {
            // get probability from gaussian equation
            float gauss_val = (1./(2.*PI*pow(sigma, 2.))) * exp(-(pow(i,2.)+pow(j,2.))/(2.*pow(sigma,2.)));
            gauss_sum += gauss_val;

            int2 coord = {x+i, y+j};
            // if coord is out of image refer the pixel on the edge
            if (coord.x < 0)  coord.x = 0;
            if (coord.y < 0) coord.y = 0;
            if (coord.x >= cols) coord.x = cols - 1;
            if (coord.y >= rows) coord.y = rows - 1;

            rgb_sum.x += gauss_val * src(coord.y, coord.x).x;
            rgb_sum.y += gauss_val * src(coord.y, coord.x).y;
            rgb_sum.z += gauss_val * src(coord.y, coord.x).z;
        }
    
    dst(y, x).x = uchar(rgb_sum.x / gauss_sum);
    dst(y, x).y = uchar(rgb_sum.y / gauss_sum);
    dst(y, x).z = uchar(rgb_sum.z / gauss_sum);
}

__device__ void calc_mean_rgb(const cv::cuda::PtrStep<uchar3> src,
                              int rows, int cols,
                              int x, int y,
                              int neighbour, 
                              float mean[3])
{
    // for rgb
    for (int n=0; n<3; n++)
    {
        // for all neighbours
        for (int j=-neighbour/2; j<neighbour/2+neighbour%2; j++)
            for (int i=-neighbour/2; i<neighbour/2+neighbour%2; i++)
            {
                int2 coord = {x+i, y+j};
                // if coord is out of edge
                if (coord.x < 0) coord.x = 0;
                if (coord.y < 0) coord.y = 0;
                if (coord.x >= cols) coord.x = cols - 1;
                if (coord.y >= rows) coord.y = rows - 1;

                uchar col;
                switch (n)
                {
                    case 0: col = src(coord.y, coord.x).x; break;
                    case 1: col = src(coord.y, coord.x).y; break;
                    case 2: col = src(coord.y, coord.x).z; break;
                }
                mean[n] += col;
            }
        mean[n] /= neighbour * neighbour;
    }
}

__device__ void calc_covariance_mat_rgb(const cv::cuda::PtrStep<uchar3> src,
                                        int rows, int cols,
                                        int x, int y,
                                        int neighbour,
                                        float covariance_mat[3][3])
{
    float mean[3] = {0., 0., 0.};
    calc_mean_rgb(src, rows, cols, x, y, neighbour, mean);

    // for all combinations of rgb
    for (int n2=0; n2<3; n2++)
        for (int n1=0; n1<3; n1++)
        {
            // for all neighbours
            for (int j=-neighbour/2; j<neighbour/2+neighbour%2; j++)
                for (int i=-neighbour/2; i<neighbour/2+neighbour%2; i++)
                {
                    int2 coord = {x+i, y+j};
                    // if coord is out of edge
                    if (coord.x < 0) coord.x = 0;
                    if (coord.y < 0) coord.y = 0;
                    if (coord.x >= cols) coord.x = cols - 1;
                    if (coord.y >= rows) coord.y = rows - 1;

                    uchar col1, col2;
                    switch (n1)
                    {
                        case 0: col1 = uchar(src(coord.y, coord.x).x); break;
                        case 1: col1 = uchar(src(coord.y, coord.x).y); break;
                        case 2: col1 = uchar(src(coord.y, coord.x).z); break;
                    }
                    switch (n2)
                    {
                        case 0: col2 = uchar(src(coord.y, coord.x).x); break;
                        case 1: col2 = uchar(src(coord.y, coord.x).y); break;
                        case 2: col2 = uchar(src(coord.y, coord.x).z); break;
                    }
                    // printf("%d, %d\n", col1, col2);
                    covariance_mat[n2][n1] += (col1-mean[n1]) * (col2-mean[n2]);
                }
            covariance_mat[n2][n1] /= neighbour * neighbour;
        }
}

__device__ float calc_determinant(const float mat[3][3])
{
    float determinant = 0.0;
    for (int n=0; n<3; n++)
    {
        determinant += mat[0][n] * (
            mat[1][(n+1)%3]*mat[2][(n+2)%3] - mat[1][(n+2)%3]*mat[2][(n+1)%3]
        );
    }
    return determinant;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst,
                        int rows, int cols,
                        int neighbour, float max_kernel, float gamma, int mode)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    float covariance_mat[3][3] = {
        0., 0., 0.,
        0., 0., 0.,
        0., 0., 0.
    };
    calc_covariance_mat_rgb(src, rows, cols, dst_x, dst_y, neighbour, covariance_mat);

    float determinant = 0.;
    determinant = calc_determinant(covariance_mat);
    determinant = log10(abs(determinant)+1); // take absolute, then convert into log10 scale


    int kernel_size = max_kernel / float(pow(determinant, gamma) + 1.);
    kernel_size = max(1, kernel_size); // kernel size must be at least 1
    if (mode == 1) { // visualize kernel size map
        dst(dst_y, dst_x).x = uchar(0);
        dst(dst_y, dst_x).y = uchar(kernel_size * 255/max_kernel);
        dst(dst_y, dst_x).z = uchar(0);
        return;
    }

    gaussian_filtering(src, dst, rows, cols, kernel_size, SIGMA, dst_x, dst_y);
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int neighbour, float max_kernel, float gamma, int mode)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, neighbour, max_kernel, gamma, mode);
}