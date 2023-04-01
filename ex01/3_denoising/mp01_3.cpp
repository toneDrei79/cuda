#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock


#define MAX_KERNEL 9
#define SIGMA 1.5
#define PI 3.1415

using namespace std;

void gaussian_filtering(const cv::Mat& src, cv::Mat& dst, int rows, int cols, int kernel_size, int x, int y)
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
            float gauss_val = (1./(2.*PI*pow(SIGMA, 2.))) * exp(-(pow(i,2.)+pow(j,2.))/(2.*pow(SIGMA,2.)));
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

void denoising(const cv::Mat& src, cv::Mat& dst, int rows, int cols, int neighbour, float gamma, int x, int y)
{
    int N = neighbour * neighbour;

    float mean[3] = {0., 0., 0.};
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
            }
        mean[n] /= N;
    }

    float covariance_mat[3][3] = {
        0., 0., 0.,
        0., 0., 0.,
        0., 0., 0.
    };
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
                    covariance_mat[n2][n1] += (src.at<cv::Vec3b>(idx_y, idx_x)[n1]-mean[n1]) * (src.at<cv::Vec3b>(idx_y, idx_x)[n1]-mean[n2]);
                }
            covariance_mat[n2][n1] /= N;
        }
    
    // cout << covariance_mat[0][0] << ' ' << covariance_mat[0][1] << ' ' << covariance_mat[0][2] << endl;
    // cout << covariance_mat[1][0] << ' ' << covariance_mat[1][1] << ' ' << covariance_mat[1][2] << endl;
    // cout << covariance_mat[2][0] << ' ' << covariance_mat[2][1] << ' ' << covariance_mat[2][2] << endl << endl;

    float determinant = 0.;
    for (int n=0; n<3; n++)
    {
        determinant += covariance_mat[0][n] * (
            covariance_mat[1][(n+1)%3]*covariance_mat[2][(n+2)%3] - covariance_mat[1][(n+2)%3]*covariance_mat[2][(n+1)%3]
        );
    }
    determinant = abs(determinant);

    int kernel_size = MAX_KERNEL / (pow(determinant, gamma) + 1);
    kernel_size = max(1, kernel_size); // kernel size must be at least 1

    // cout << determinant << " : ";
    // cout << kernel_size << endl;

    gaussian_filtering(src, dst, rows, cols, kernel_size, x, y);
}

int main(int argc, char** argv)
{
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::Mat h_img = cv::imread(argv[2]);
    cv::Mat h_result(h_img.rows, h_img.cols, CV_8UC3);

    cv::imshow("Original Image", h_img);

    int neighbour = std::stoi(argv[3]);
    float gamma = std::stof(argv[4]);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = std::stoi(argv[1]);
    for (int i=0; i<iter ;i++)
    {
        #pragma omp parallel for
            // for each pixel
            for (int j=0; j<h_result.rows; j++)
                for (int i=0; i<h_result.cols; i++)
                {
                    denoising(h_img, h_result, h_result.rows, h_result.cols, neighbour, gamma, i, j);
                }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", h_result);

    cout << "Time: " << diff.count() << endl;
    cout << "Time/frame: " << diff.count()/iter << endl;
    cout << "IPS: " << iter/diff.count() << endl;

    cv::waitKey();

    return 0;
}