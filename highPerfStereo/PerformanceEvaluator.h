//
// Created by dallas on 13.05.17.
//

#ifndef PROJECT_PERFORMANCEEVALUATOR_H
#define PROJECT_PERFORMANCEEVALUATOR_H

#include <opencv2/opencv.hpp>

using namespace cv;

class PerformanceEvaluator {

    Mat kinectRawDepth; // Raw depth calculated by the kinect
    std::vector<Point3f> kinectPointCloud; // 3D points obtained from the raw depth, in the reference frame of the kinect
    std::vector<Point2f> kinectPointsInImage; // Pixel coordinates of the points from the kinect inside the image for which the disparity map has been calculated

    Mat disparities; // Disparity map
    vector<Point> consideredDisparities; // Pixels for which there is a disparity values
    std::vector<Point3f> stereoPointCloud; // 3D points obtained from disparity map

    Mat camMatrix; // Camera matrix
    Mat perspTransform; // Perspective transformation matrix
    Mat R; // Rotation matrix from the camera of the stereo system to the camera of the kinect
    Vec3d T; // translation vector from the camera of the stereo system to the camera of the kinect



};


#endif //PROJECT_PERFORMANCEEVALUATOR_H
