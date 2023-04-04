#pragma once
#include <opencv2/opencv.hpp>


void calc_mean_rgb(const cv::Mat& src, int rows, int cols, int x, int y, int neighbour, float mean[3]);

void calc_covariance_mat_rgb(const cv::Mat& src, int rows, int cols, int x, int y, int neighbour, float covariance_mat[3][3]);

float calc_determinant(const float mat[3][3]);