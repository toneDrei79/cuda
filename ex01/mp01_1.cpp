#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock

#include "anaglyph_mats.h"


using namespace std;

void anaglyph(const cv::Mat& src, cv::Mat& dst, int rows, int cols, float mat_l[3][3], float mat_r[3][3])
{
    #pragma omp parallel for 
        for (int j=0; j<dst.rows; j++)
            for (int i=0; i<dst.cols; i++)
            {
                uchar lr = src.at<cv::Vec3b>(j,i)[2];
                uchar lg = src.at<cv::Vec3b>(j,i)[1];
                uchar lb = src.at<cv::Vec3b>(j,i)[0];
                uchar rr = src.at<cv::Vec3b>(j,i+cols)[2];
                uchar rg = src.at<cv::Vec3b>(j,i+cols)[1];
                uchar rb = src.at<cv::Vec3b>(j,i+cols)[0];


                dst.at<cv::Vec3b>(j,i)[0] = 0;
                dst.at<cv::Vec3b>(j,i)[1] = 0;
                dst.at<cv::Vec3b>(j,i)[2] = 255;

                // uchar pixel = src.at<uchar>(j,i);
                // cout << pixel;
                // break;
                // dst(j,i).z = char(mat_l[0][0]*left.z + mat_l[0][1]*left.y + mat_l[0][2]*left.x + mat_r[0][0]*right.z + mat_r[0][1]*right.y + mat_r[0][2]*right.x);
                // dst(j,i).y = char(mat_l[1][0]*left.z + mat_l[1][1]*left.y + mat_l[1][2]*left.x + mat_r[1][0]*right.z + mat_r[1][1]*right.y + mat_r[1][2]*right.x);
                // dst(j,i).x = char(mat_l[2][0]*left.z + mat_l[2][1]*left.y + mat_l[2][2]*left.x + mat_r[2][0]*right.z + mat_r[2][1]*right.y + mat_r[2][2]*right.x);
            }

        
}

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
        anaglyph(h_img, h_result, h_result.rows, h_result.cols, mat_l, mat_r);
        break;
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