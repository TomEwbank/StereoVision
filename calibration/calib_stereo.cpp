#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

vector< vector< Point3f > > object_points;
vector< vector< Point2f > > imagePoints1, imagePoints2;
vector< Point2f > corners1, corners2;
vector< vector< Point2f > > left_img_points, right_img_points;

Mat img1, img2, gray1, gray2;

void load_image_points(int board_width, int board_height, int num_imgs, float square_size,
                       string leftimg_dir, string rightimg_dir, string leftimg_filename, string rightimg_filename) {

    Size board_size = Size(board_width, board_height);
    int board_n = board_width * board_height;

    for (int i = 1; i <= num_imgs; i++) {
        string left_img = leftimg_dir+leftimg_filename+to_string(i)+".jpg";
        string right_img = rightimg_dir+rightimg_filename+to_string(i)+".jpg";
        img1 = imread(left_img, CV_LOAD_IMAGE_COLOR);
        img2 = imread(right_img, CV_LOAD_IMAGE_COLOR);
        cvtColor(img1, gray1, CV_BGR2GRAY);
        cvtColor(img2, gray2, CV_BGR2GRAY);

        bool found1 = cv::findChessboardCorners(img1, board_size, corners1,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        bool found2 = cv::findChessboardCorners(img2, board_size, corners2,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

        if (found1) {
            cv::cornerSubPix(gray1, corners1, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray1, board_size, corners1, found1);
        }
        if (found2) {
            cv::cornerSubPix(gray2, corners2, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
            cv::drawChessboardCorners(gray2, board_size, corners2, found2);
        }

        vector<Point3f> obj;
        for (int i = 0; i < board_height; i++)
            for (int j = 0; j < board_width; j++)
                obj.push_back(Point3f((float) j * square_size, (float) i * square_size, 0));

        if (found1 && found2) {
            cout << i << ". Found corners!" << endl;
            imagePoints1.push_back(corners1);
            imagePoints2.push_back(corners2);
            object_points.push_back(obj);
        }
    }
    for (int i = 0; i < imagePoints1.size(); i++) {
        vector<Point2f> v1, v2;
        for (int j = 0; j < imagePoints1[i].size(); j++) {
            v1.push_back(Point2f((double) imagePoints1[i][j].x, (double) imagePoints1[i][j].y));
            v2.push_back(Point2f((double) imagePoints2[i][j].x, (double) imagePoints2[i][j].y));
        }
        left_img_points.push_back(v1);
        right_img_points.push_back(v2);
    }
}

double computeReprojectionErrors(const vector< vector< Point3f > >& objectPoints,
                                 const vector< vector< Point2f > >& imagePoints,
                                 const vector< Mat >& rvecs, const vector< Mat >& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs) {
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    vector<float> perViewErrors;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int) objectPoints.size(); ++i) {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
                      distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
        int n = (int) objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }
    return std::sqrt(totalErr / totalPoints);
}

int main(int argc, char const *argv[])
{
    int board_width = 9;
    int board_height = 6;
    float square_size = 0.02423;
    string leftimg_dir = "calib_imgs/1/";
    string rightimg_dir = "calib_imgs/1/";
    string leftimg_filename = "left";
    string rightimg_filename = "right";
    string out_file = "stereoCalib.yml";
    int num_imgs = 29;

    load_image_points(board_width, board_height, num_imgs, square_size,
               leftimg_dir, rightimg_dir, leftimg_filename, rightimg_filename);

    printf("Starting Calibration\n");
    Mat K1, K2;
    Mat D1, D2;
    vector< Mat > rvecs1, tvecs1;
    vector< Mat > rvecs2, tvecs2;
    int flag = 0;
    flag |= CV_CALIB_FIX_K4;
    flag |= CV_CALIB_FIX_K5;

    calibrateCamera(object_points, imagePoints1, img1.size(), K1, D1, rvecs1, tvecs1, flag);
    cout << "Calibration error cam1: " << computeReprojectionErrors(object_points, imagePoints1, rvecs1, tvecs1, K1, D1) << endl;

    calibrateCamera(object_points, imagePoints2, img2.size(), K2, D2, rvecs2, tvecs2, flag);
    cout << "Calibration error cam2: " << computeReprojectionErrors(object_points, imagePoints2, rvecs2, tvecs2, K2, D2) << endl;

    Mat  R, F, E;
    Vec3d T;
    stereoCalibrate(object_points, left_img_points, right_img_points, K1, D1, K2, D2, img1.size(), R, T, E, F);

    cv::FileStorage fs1(out_file, cv::FileStorage::WRITE);
    fs1 << "K1" << K1;
    fs1 << "K2" << K2;
    fs1 << "D1" << D1;
    fs1 << "D2" << D2;
    fs1 << "R" << R;
    fs1 << "T" << T;
    fs1 << "E" << E;
    fs1 << "F" << F;

    printf("Done Calibration\n");

    printf("Starting Rectification\n");

    cv::Mat R1, R2, P1, P2, Q;
    stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);

    fs1 << "R1" << R1;
    fs1 << "R2" << R2;
    fs1 << "P1" << P1;
    fs1 << "P2" << P2;
    fs1 << "Q" << Q;

    printf("Done Rectification\n");

    return 0;
}
