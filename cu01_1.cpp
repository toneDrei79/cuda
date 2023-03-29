#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock

#include "anaglyphs.h"


using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

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

    int choice = stoi(argv[2]);
    float matL[3][3];
    float matR[3][3];
    switch (choice) {
    case 0: // true
        matL = trueL;
        matR = trueR;
        break;
    case 1: // gray
        matL = grayL;
        matR = grayR;
        break;
    case 2: // color
        matL = colorL;
        matR = colorR;
        break;
    case 3: // halfcolor
        matL = halfcolorL;
        matR = halfcolorR;
        break;
    case 4: // optimized
        matL = optimizedL;
        matR = optimizedR;
        break;
    default:
    }

    cout << matL << endl;
    cout << matL << endR;

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100000;
    for (int i=0; i<iter ;i++)
    {
        startCUDA(d_img, d_result);
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