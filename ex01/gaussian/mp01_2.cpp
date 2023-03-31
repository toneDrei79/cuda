#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock


#define PI 3.1415
using namespace std;

void gaussian_filtering(const cv::Mat& src, cv::Mat& dst, int rows, int cols, int kernel_size, float sigma, int x, int y)
{
    float r_sum = 0;
    float g_sum = 0;
    float b_sum = 0;
    float gauss_sum = 0;
    for (int j=-kernel_size/2; j<kernel_size/2+kernel_size%2; j++)
        for (int i=-kernel_size/2; i<kernel_size/2+kernel_size%2; i++)
        {
            float gauss_val = (1./(2.*PI*pow(sigma, 2.))) * exp(-(pow(i,2.)+pow(j,2.))/(2.*pow(sigma,2.)));
            gauss_sum += gauss_val;

            int idx_x = x + i;
            int idx_y = y + j;
            if (idx_x < 0) idx_x = 0;
            if (idx_y < 0) idx_y = 0;
            if (idx_x >= cols) idx_x = cols - 1;
            if (idx_y >= rows) idx_y = rows - 1;

            r_sum += gauss_val * src.at<cv::Vec3b>(idx_y, idx_x)[2];
            g_sum += gauss_val * src.at<cv::Vec3b>(idx_y, idx_x)[1];
            b_sum += gauss_val * src.at<cv::Vec3b>(idx_y, idx_x)[0];
        }
    
    dst.at<cv::Vec3b>(y,x)[2] = uchar(r_sum / gauss_sum);
    dst.at<cv::Vec3b>(y,x)[1] = uchar(g_sum / gauss_sum);
    dst.at<cv::Vec3b>(y,x)[0] = uchar(b_sum / gauss_sum);
}

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[1]);
    cv::Mat h_result(h_img.rows, h_img.cols, CV_8UC3);

    cv::imshow("Original Image", h_img);

    int kernel_size = std::stoi(argv[2]);
    float sigma = std::stof(argv[3]);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 1;
    for (int i=0; i<iter ;i++)
    {
        #pragma omp parallel for 
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