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

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int neighbour, float gamma)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;





    int N = neighbour * neighbour;

    float mean[3] = {0., 0., 0.};
    // for rgb
    for (int n=0; n<3; n++)
    {
        // for all neighbours
        for (int j=-neighbour/2; j<neighbour/2+neighbour%2; j++)
            for (int i=-neighbour/2; i<neighbour/2+neighbour%2; i++)
            {
                int2 coord = {dst_x+i, dst_y+j};
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
                // mean[n] += src(coord.y, coord.x)[n];
            }
        mean[n] /= N;
    }
    // printf("%f %f %f\n", mean[0], mean[1], mean[2]);
    // dst(dst_y, dst_x).x = uchar(mean[0]);
    // dst(dst_y, dst_x).y = uchar(mean[1]);
    // dst(dst_y, dst_x).z = uchar(mean[2]);    
    // return;

    float covariance_mat[3][3] = {
        0., 0., 0.,
        0., 0., 0.,
        0., 0., 0.
    };
    // printf("%ld\n", sizeof(covariance_mat[0][0]));
    // for all combinations of rgb
    for (int n2=0; n2<3; n2++)
        for (int n1=0; n1<3; n1++)
        {
            // for all neighbours
            for (int j=-neighbour/2; j<neighbour/2+neighbour%2; j++)
                for (int i=-neighbour/2; i<neighbour/2+neighbour%2; i++)
                {
                    int2 coord = {dst_x+i, dst_y+j};
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
            covariance_mat[n2][n1] /= N;
        }
    
    // dst(dst_y, dst_x).x = uchar(covariance_mat[0][0]);
    // dst(dst_y, dst_x).y = uchar(covariance_mat[0][0]);
    // dst(dst_y, dst_x).z = uchar(covariance_mat[0][0]);
    // return;
    
    // printf("%f %f %f\n", covariance_mat[0][0], covariance_mat[0][1], covariance_mat[0][2]);
    // printf("%f %f %f\n", covariance_mat[1][0], covariance_mat[1][1], covariance_mat[1][2]);
    // printf("%f %f %f\n\n", covariance_mat[2][0], covariance_mat[2][1], covariance_mat[2][2]);

    float determinant = 0.;
    for (int n=0; n<3; n++)
    {
        determinant += covariance_mat[0][n] * (
            covariance_mat[1][(n+1)%3]*covariance_mat[2][(n+2)%3] - covariance_mat[1][(n+2)%3]*covariance_mat[2][(n+1)%3]
        );
    }
    determinant = log10(abs(determinant)+1);
    // printf("%f\n", determinant);


    // printf("%f\n", determinant);


    int kernel_size = MAX_KERNEL / float(pow(determinant, gamma) + 1.);
    kernel_size = max(1, kernel_size); // kernel size must be at least 1
    // dst(dst_y, dst_x).x = uchar(0);
    // dst(dst_y, dst_x).y = uchar(kernel_size * 15);
    // dst(dst_y, dst_x).z = uchar(0);
    // return;

    // printf("%d\n", kernel_size);

    float3 rgb_sum = {0, 0, 0};
    float gauss_sum = 0;
    // for each kernel pixel
    for (int j=-kernel_size/2; j<kernel_size/2+kernel_size%2; j++)
        for (int i=-kernel_size/2; i<kernel_size/2+kernel_size%2; i++)
        {
            // get probability from gaussian equation
            float gauss_val = (1./(2.*PI*pow(SIGMA, 2.))) * exp(-(pow(i,2.)+pow(j,2.))/(2.*pow(SIGMA,2.)));
            gauss_sum += gauss_val;

            int2 coord = {dst_x+i, dst_y+j};
            // if coord is out of edge
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

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int neighbour, float gamma)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

    process<<<grid, block>>>(src, dst, dst.rows, dst.cols, neighbour, gamma);
}