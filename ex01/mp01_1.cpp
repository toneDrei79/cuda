#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

#include "anaglyphs.h"


using namespace std;

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[1]);
    cv::Mat h_result(h_img.rows, h_img.cols/2, CV_8UC3);

    cv::imshow("Original Image", h_img);

    int choice = std::stoi(argv[2]); // choose anaglyph mode
    // matrices for anaglyph
    float mat_l[3][3];
    float mat_r[3][3];
    set_anaglyph_mats(choice, mat_l, mat_r);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 100000;
    for (int i=0; i<iter ;i++)
    {
        #pragma omp parallel for

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