#include "gaussian_filtering.h"


void gaussian_filtering(const cv::Mat& src, cv::Mat& dst,
                        int rows, int cols,
                        int kernel_size, float sigma,
                        int x, int y)
{
    float r_sum = 0;
    float g_sum = 0;
    float b_sum = 0;
    float gauss_sum = 0;
    // for each kernel pixel
    for (int j=-kernel_size/2; j<kernel_size/2+kernel_size%2; j++)
        for (int i=-kernel_size/2; i<kernel_size/2+kernel_size%2; i++)
        {
            // get probability from gaussian equation
            float gauss_val = (1./(2.*PI*pow(sigma, 2.))) * exp(-(pow(i,2.)+pow(j,2.))/(2.*pow(sigma,2.)));
            gauss_sum += gauss_val;

            int idx_x = x + i;
            int idx_y = y + j;
            // if idx is out of edge
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