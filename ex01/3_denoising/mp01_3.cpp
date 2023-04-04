#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

#include "calc_covariance.h"
#include "gaussian_filtering.h"


#define SIGMA 1.5

using namespace std;

void denoising(const cv::Mat& src, cv::Mat& dst,
               int rows, int cols,
               int neighbour, float max_kernel, float gamma, int mode,
               int x, int y)
{
    float covariance_mat[3][3] = {
        0., 0., 0.,
        0., 0., 0.,
        0., 0., 0.
    };
    calc_covariance_mat_rgb(src, rows, cols, x, y, neighbour, covariance_mat);
    
    float determinant = 0.;
    determinant = calc_determinant(covariance_mat);
    determinant = log10(abs(determinant)+1); // take absolute, then convert into log10 scale


    int kernel_size = max_kernel / (pow(determinant, gamma) + 1.);
    kernel_size = max(1, kernel_size); // kernel size must be at least 1
    if (mode == 1) { // visualize kernel size map
        dst.at<cv::Vec3b>(y,x)[0] = uchar(0);
        dst.at<cv::Vec3b>(y,x)[1] = uchar(kernel_size * 255/max_kernel);
        dst.at<cv::Vec3b>(y,x)[2] = uchar(0);
        return;
    }

    gaussian_filtering(src, dst, rows, cols, kernel_size, SIGMA, x, y);
}

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[2]);
    cv::Mat h_result(h_img.rows, h_img.cols, CV_8UC3);

    cv::imshow("Original Image", h_img);

    int neighbour = std::stoi(argv[3]);
    float max_kernel = std::stof(argv[4]);
    float gamma = std::stof(argv[5]);
    int mode = std::stoi(argv[6]); // 0 -> visualize gaussian filtered image, 1 -> visualize kernel size map

    auto begin = chrono::high_resolution_clock::now();
    const int iter = std::stoi(argv[1]);
    for (int i=0; i<iter ;i++)
    {
        #pragma omp parallel for
            // for each pixel
            for (int j=0; j<h_result.rows; j++)
                for (int i=0; i<h_result.cols; i++)
                {
                    denoising(h_img, h_result, h_result.rows, h_result.cols, neighbour, max_kernel, gamma, mode, i, j);
                }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", h_result);

    cout << "Time: " << diff.count() << endl;
    cout << "Time/frame: " << diff.count()/iter << endl;
    cout << "IPS: " << iter/diff.count() << endl;

    cv::waitKey();

    return 0;
}