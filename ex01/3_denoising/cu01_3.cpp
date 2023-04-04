#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>  // for high_resolution_clock


using namespace std;

void startCUDA(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int kernel_size, float max_kernel, float gamma, int mode);

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[2]);
    cv::Mat h_result(h_img.rows, h_img.cols, CV_8UC3);

    cv::cuda::GpuMat d_img, d_result;
    d_img.upload(h_img);

    cv::imshow("Original Image", h_img);

    int neighbour_size = std::stoi(argv[3]);
    float max_kernel = std::stof(argv[4]);
    float gamma = std::stof(argv[5]);
    int mode = std::stoi(argv[6]); // 0 -> visualize gaussian filtered image, 1 -> visualize kernel size map

    auto begin = chrono::high_resolution_clock::now();
    const int iter = std::stoi(argv[1]);
    for (int i=0; i<iter ;i++)
    {
        d_result.upload(h_result);
        startCUDA(d_img, d_result, neighbour_size, max_kernel, gamma, mode);
        d_result.download(h_result);
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