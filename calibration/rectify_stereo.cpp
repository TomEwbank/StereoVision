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
    String inputFolder = "imgs_to_rectify/";
    String outputFolder = "imgs_rectified/";
    String calibFile = inputFolder+"stereoParams_2906.yml";//"stereoCalib_2305_rotx008_nothingInv.yml";//"stereoParams_2806_windows_camUntouched.yml";//
    String pairName = "1_200_floor_inclination";
    String imageExtension = ".ppm";

    int imHeight = 460;
    int imWidth = 800;
//    double sizeFactor = 1;
    Size size(imWidth, imHeight);
//    Size newSize(imWidth*sizeFactor,imHeight*sizeFactor);
    Size newSize(1037,631);
    int imgsNumber = 4;

    Mat K1, K2;
    Mat D1, D2;
    Mat  R, F, E;
    Vec3d T;

    FileStorage fs;
    fs.open(calibFile, FileStorage::READ);

    fs["K1"] >> K1;
    fs["K2"] >> K2;
    fs["D1"] >> D1;
    fs["D2"] >> D2;
    fs["R"] >> R;
    fs["T"] >> T;
    
//    Mat Q,P1,R1,P2,R2;
//    fs["Q"] >> Q;
//    fs["R1"] >> R1;
//    fs["R2"] >> R2;
//    fs["P1"] >> P1;
//    fs["P2"] >> P2;

    fs.release();

    cv::Mat R1, R2, P1, P2, Q;
    Rect validPixROI1(0,0,0,0), validPixROI2(0,0,0,0);
    stereoRectify(K1, D1, K2, D2, size, R, T, R1, R2, P1, P2, Q, 0, -1, newSize, &validPixROI1, &validPixROI2);
//    cout << validPixROI1.x << endl;
//    cout << validPixROI1.y << endl;
//    cout << validPixROI1.width << endl;
//    cout << validPixROI1.height << endl;

    cv::FileStorage fs1(calibFile, cv::FileStorage::WRITE);
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

    fs1.release();


    // Rectify the images
    for (int i=1; i <= imgsNumber; ++i) {

        String imNumber = std::to_string(i);
        Mat img1 = imread(inputFolder+"left_"+pairName+"_"+imNumber+imageExtension, CV_LOAD_IMAGE_COLOR);
        Mat img2 = imread(inputFolder+"right_"+pairName+"_"+imNumber+imageExtension, CV_LOAD_IMAGE_COLOR);

        cv::Mat lmapx, lmapy, rmapx, rmapy;
        cv::Mat imgU1, imgU2;

        cv::initUndistortRectifyMap(K1, D1, R1, P1, newSize, CV_32F, lmapx, lmapy);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, newSize, CV_32F, rmapx, rmapy);
        cv::remap(img1, imgU1, lmapx, lmapy, cv::INTER_LINEAR);
        cv::remap(img2, imgU2, rmapx, rmapy, cv::INTER_LINEAR);

        imwrite(outputFolder + "left_" + pairName + "_" + imNumber + "_rectified.png", imgU1);
        imwrite(outputFolder + "right_" + pairName + "_" + imNumber + "_rectified.png", imgU2);
    }

    return 0;
}
