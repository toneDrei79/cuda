#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

#include "gaussian_filtering.h"


using namespace std;

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[2]);
    cv::Mat h_result(h_img.rows, h_img.cols, CV_8UC3);

    cv::imshow("Original Image", h_img);

    int kernel_size = std::stoi(argv[3]);
    float sigma = std::stof(argv[4]);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = std::stoi(argv[1]);
    for (int i=0; i<iter ;i++)
    {
        #pragma omp parallel for 
            // for each pixel
            for (int j=0; j<h_result.rows; j++)
                for (int i=0; i<h_result.cols; i++)
                {
                    gaussian_filtering(h_img, h_result, h_result.rows, h_result.cols, kernel_size, sigma, i, j);
                }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", h_result);

    cout << "Time: "<< diff.count() << endl;
    cout << "Time/frame: " << diff.count()/iter << endl;
    cout << "IPS: " << iter/diff.count() << endl;

    cv::waitKey();

    return 0;
}