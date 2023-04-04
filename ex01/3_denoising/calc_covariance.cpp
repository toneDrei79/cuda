#include "calc_covariance.h"


void calc_mean_rgb(const cv::Mat& src,
                   int rows, int cols,
                   int x, int y,
                   int neighbour, 
                   float mean[3])
{
    // for rgb
    for (int n=0; n<3; n++)
    {
        // for all neighbours
        for (int j=-neighbour/2; j<neighbour/2+neighbour%2; j++)
            for (int i=-neighbour/2; i<neighbour/2+neighbour%2; i++)
            {
                int idx_x = x + i;
                int idx_y = y + j;
                // if idx is out of edge
                if (idx_x < 0) idx_x = 0;
                if (idx_y < 0) idx_y = 0;
                if (idx_x >= cols) idx_x = cols - 1;
                if (idx_y >= rows) idx_y = rows - 1;
                mean[n] += src.at<cv::Vec3b>(idx_y, idx_x)[n];
                int a = src.at<cv::Vec3b>(idx_y, idx_x)[n];
                // cout << a << endl;
            }
        mean[n] /= neighbour * neighbour;
    }
}

void calc_covariance_mat_rgb(const cv::Mat& src,
                             int rows, int cols,
                             int x, int y,
                             int neighbour,
                             float covariance_mat[3][3])
{
    float mean[3] = {0., 0., 0.};
    calc_mean_rgb(src, rows, cols, x, y, neighbour, mean);
    
    // for all combinations of rgb
    for (int n2=0; n2<3; n2++)
        for (int n1=0; n1<3; n1++)
        {
            // for all neighbours
            for (int j=-neighbour/2; j<neighbour/2+neighbour%2; j++)
                for (int i=-neighbour/2; i<neighbour/2+neighbour%2; i++)
                {
                    int idx_x = x + i;
                    int idx_y = y + j;
                    // if idx is out of edge
                    if (idx_x < 0) idx_x = 0;
                    if (idx_y < 0) idx_y = 0;
                    if (idx_x >= cols) idx_x = cols - 1;
                    if (idx_y >= rows) idx_y = rows - 1;
                    covariance_mat[n2][n1] += (src.at<cv::Vec3b>(idx_y, idx_x)[n1]-mean[n1]) * (src.at<cv::Vec3b>(idx_y, idx_x)[n2]-mean[n2]);
                    int a = src.at<cv::Vec3b>(idx_y, idx_x)[n1];
                    int b = src.at<cv::Vec3b>(idx_y, idx_x)[n2];
                    // cout << a << ", " << b << endl;
                }
            covariance_mat[n2][n1] /= neighbour * neighbour;
        }
}

float calc_determinant(const float mat[3][3])
{
    float determinant = 0.0;
    for (int n=0; n<3; n++)
    {
        determinant += mat[0][n] * (
            mat[1][(n+1)%3]*mat[2][(n+2)%3] - mat[1][(n+2)%3]*mat[2][(n+1)%3]
        );
    }
    return determinant;
}