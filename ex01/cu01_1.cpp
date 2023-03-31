#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

#include "anaglyphs.h"


using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float* mat_l, float* mat_r);

void set_anaglyph_mats(int choice, float mat_l[3][3], float mat_r[3][3]);

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[1]);
    cv::Mat h_result(h_img.rows, h_img.cols/2, CV_8UC3);

    cv::cuda::GpuMat d_img, d_result;

    d_img.upload(h_img);
    d_result.upload(h_result);

    cv::imshow("Original Image", h_img);

    int choice = std::stoi(argv[2]);
    float mat_l[3][3];
    float mat_r[3][3];
    set_anaglyph_mats(choice, mat_l, mat_r);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100000;
    for (int i=0; i<iter ;i++)
    {
        startCUDA(d_img, d_result, &mat_l[0][0], &mat_r[0][0]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    d_result.download(h_result);
    cv::imshow("Processed Image", h_result);

    cout << "Time: "<< diff.count() << endl;
    cout << "Time/frame: " << diff.count()/iter << endl;
    cout << "IPS: " << iter/diff.count() << endl;

    cv::waitKey();

    return 0;
}

void set_anaglyph_mats(int choice, float mat_l[3][3], float mat_r[3][3])
{
    switch (choice)
    {
        case 0: set_true_mats(mat_l, mat_r); break;
        case 1: set_gray_mats(mat_l, mat_r); break;
        case 2: set_color_mats(mat_l, mat_r); break;
        case 3: set_halfcolor_mats(mat_l, mat_r); break;
        case 4: set_optimized_mats(mat_l, mat_r); break;
        default:;
    }
}