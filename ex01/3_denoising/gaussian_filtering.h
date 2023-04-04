#pragma once
#include <opencv2/opencv.hpp>

#define PI 3.1415

void gaussian_filtering(const cv::Mat& src, cv::Mat& dst, int rows, int cols, int kernel_size, float sigma, int x, int y);