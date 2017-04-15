#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
    Mat disparities = imread("img/left_1_500_01_rectified2_disp.pgm", CV_LO);
    cout << disparities.at(Point(0,0));
    return 0;
}