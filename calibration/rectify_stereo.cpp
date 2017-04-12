#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dirent.h>

using namespace std;
using namespace cv;


int main(int argc, char const *argv[])
{
    String out_file = "stereoMatlabCalib.yml";
    Mat img1 = imread("left_1_500_01.ppm", CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread("right_1_500_01.ppm", CV_LOAD_IMAGE_COLOR);

    Mat K1, K2;
    Mat D1, D2;
    Mat  R, F, E;
    Vec3d T;

    FileStorage fs;
    fs.open(out_file, FileStorage::READ);

    fs["K1"] >> K1;
    fs["K2"] >> K2;
    fs["D1"] >> D1;
    fs["D2"] >> D2;
    fs["R"] >> R;
    fs["T"] >> T;

    fs.release();

    printf("Starting Rectification\n");

    cv::Mat R1, R2, P1, P2, Q;
    Rect validPixROI1(0,0,0,0), validPixROI2(0,0,0,0);
    Size newSize(img1.cols,img1.rows);//(img1.size().width*1.5,img1.size().height*1.2);
    stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, Q, 0, -1, newSize, &validPixROI1, &validPixROI2);
    cout << validPixROI1.x << endl;
    cout << validPixROI1.y << endl;
    cout << validPixROI1.width << endl;
    cout << validPixROI1.height << endl;

    cv::FileStorage fs1(out_file, cv::FileStorage::WRITE);
    fs1 << "K1" << K1;
    fs1 << "K2" << K2;
    fs1 << "D1" << D1;
    fs1 << "D2" << D2;
    fs1 << "R" << R;
    fs1 << "T" << T;
    fs1 << "R1" << R1;
    fs1 << "R2" << R2;
    fs1 << "P1" << P1;
    fs1 << "P2" << P2;
    fs1 << "Q" << Q;
    fs1 << "ROI_left" << validPixROI1;
    fs1 << "ROI_right" << validPixROI2;
    fs1 << "common_ROI" << (validPixROI1 & validPixROI2);
//    fs1 << "ROI_left_x" << validPixROI1->x;
//    fs1 << "ROI_left_y" << validPixROI1->y;
//    fs1 << "ROI_left_h" << validPixROI1->height;
//    fs1 << "ROI_left_w" << validPixROI1->width;
//    fs1 << "ROI_right_x" << validPixROI2->x;
//    fs1 << "ROI_right_y" << validPixROI1->y;
//    fs1 << "ROI_right_h" << validPixROI1->height;
//    fs1 << "ROI_right_w" << validPixROI1->width;

    fs1.release();

    printf("Done Rectification\n");

    // Rectify an image

    cv::Mat lmapx, lmapy, rmapx, rmapy;
    cv::Mat imgU1, imgU2;

    cv::initUndistortRectifyMap(K1, D1, R1, P1, newSize, CV_32F, lmapx, lmapy);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, newSize, CV_32F, rmapx, rmapy);
    cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
    cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

    imwrite("left_1_500_01_rectified.ppm", imgU1);
    imwrite("right_1_500_01_rectified.ppm", imgU2);

    Vec3d p(346,180,48);
    std::vector<Vec3d> vin(1);
    vin[0] = p;
    std::vector<Vec3d> vout(1);
    perspectiveTransform(vin,vout,Q);

    for (int i = 0; i < vout.size(); ++i) {
        cout << vin.at(i).val[0] << ", " << vin.at(i).val[1] << ", " << vin.at(i).val[2] << endl;
        cout << vout[i].val[0] << ", " << vout[i].val[1] << ", " << vout[i].val[2] << endl;
        double dist = sqrt(pow(vout[i].val[0],2) + pow(vout[i].val[1],2) + pow(vout[i].val[2],2));
        cout << dist << endl;
    }

    return 0;
}
