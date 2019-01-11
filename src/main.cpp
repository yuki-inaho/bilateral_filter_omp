#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

cv::Mat
Padding(cv::Mat img, int kernel_size)
{
    cv::Mat img_smoothed = cv::Mat::zeros(img.rows+kernel_size-1, img.cols+kernel_size-1, CV_8UC3);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            img_smoothed.at<cv::Vec3b>(y+kernel_size/2, x+kernel_size/2) = img.at<cv::Vec3b>(y, x);
        }
    }
    return img_smoothed;
}



cv::Vec3b
_BilateralFilter(cv::Mat img_padded, int x, int y, int kernel_size, double sigma_pos, double sigma_col)
{
    cv::Vec3b kernel_var(0,0,0);
    std::vector<double> _kernel_var, W;
    for(int i=0;i<3;i++){
        _kernel_var.push_back(0.0);
        W.push_back(0.0);
    }

    for(int k_y=-kernel_size/2; k_y<=kernel_size/2; k_y++){
        for(int k_x=-kernel_size/2; k_x<=kernel_size/2; k_x++){
            cv::Vec3b centor_col = img_padded.at<cv::Vec3b>(y,x);
            cv::Vec3b perf_col = img_padded.at<cv::Vec3b>(y+k_y,x+k_x);
            double diff_pos = std::sqrt(double(k_x)*double(k_x) + double(k_y)*double(k_y));
            double diff_col_r = double(centor_col[0]) - double(perf_col[0]);
            double diff_col_g = double(centor_col[1]) - double(perf_col[1]);
            double diff_col_b = double(centor_col[2]) - double(perf_col[2]);
            double diff_col = std::sqrt(diff_col_r*diff_col_r + diff_col_g*diff_col_g + diff_col_b*diff_col_b);
            double kernel_pos = std::exp(-diff_pos*diff_pos/(2*sigma_pos*sigma_pos));
            double kernel_col = std::exp(-diff_col*diff_col/(2*sigma_col*sigma_col));
            _kernel_var[0] += kernel_pos * kernel_col * double(perf_col[0]);
            _kernel_var[1] += kernel_pos * kernel_col * double(perf_col[1]);
            _kernel_var[2] += kernel_pos * kernel_col * double(perf_col[2]);
            W[0] += kernel_pos * kernel_col;
            W[1] += kernel_pos * kernel_col;
            W[2] += kernel_pos * kernel_col;
        }
    }
    
    kernel_var[0] = static_cast<unsigned char>(_kernel_var[0]/(W[0]+0.00001));
    kernel_var[1] = static_cast<unsigned char>(_kernel_var[1]/(W[1]+0.00001));
    kernel_var[2] = static_cast<unsigned char>(_kernel_var[2]/(W[2]+0.00001));

    return kernel_var;
}

cv::Mat
BilateralFilter(cv::Mat img, int kernel_size, double sigma_pos, double sigma_col)
{
    cv::Mat img_smoothed = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    cv::Mat img_padded = Padding(img, kernel_size);
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            cv::Vec3b kernel_var = _BilateralFilter(img_padded, x+kernel_size/2, y+kernel_size/2, kernel_size, sigma_pos, sigma_col);
            img_smoothed.at<cv::Vec3b>(y, x) = kernel_var;
        }
    }
    return img_smoothed;
}

cv::Mat
BilateralFilterOMP(cv::Mat img, int kernel_size, double sigma_pos, double sigma_col)
{
    cv::Mat img_smoothed = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    cv::Mat img_padded = Padding(img, kernel_size);

    cv::parallel_for_(cv::Range(0, img.rows*img.cols), [&](const cv::Range& range){
        for (int r = range.start; r < range.end; r++)
        {
            int y = r / img.cols;
            int x = r % img.cols;
            cv::Vec3b kernel_var = _BilateralFilter(img_padded, x+kernel_size/2, y+kernel_size/2, kernel_size, sigma_pos, sigma_col);
            img_smoothed.at<cv::Vec3b>(y, x) = kernel_var;
        }
    });

    return img_smoothed;
}

int main(int argc, const char * argv[]){
    cv::Mat img = cv::imread("../data/img.jpg");
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

    Mat img_smoothed = BilateralFilterOMP(img, 5, 0.1, 1);
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    cout << elapsed << endl;

    while(true){
        int key = cv::waitKey( 30 );    
        cv::imshow("img", img_smoothed);
        if ( key == 'q' ) {
            break;
        }
    }
    cv::destroyAllWindows();

}
