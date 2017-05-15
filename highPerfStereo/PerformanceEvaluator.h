//
// Created by dallas on 13.05.17.
//

#ifndef PROJECT_PERFORMANCEEVALUATOR_H
#define PROJECT_PERFORMANCEEVALUATOR_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using namespace cv;

class PerformanceEvaluator {

    float depthLookUp[2048];

    Mat kinectRawDepth; // Raw depth calculated by the kinect
    std::vector<Point3f> kinectPointCloud; // 3D points obtained from the raw depth, in the reference frame of the kinect

    int xOffset, yOffset;
    Mat disparities; // Disparity map
    std::vector<Point> consideredDisparities; // Pixels for which there is a disparity values
    std::vector<Point3f> stereoPointCloud; // 3D points obtained from disparity map
    std::vector<Point2f> stereoPointsInKinect; // Pixel coordinates of the 3D points obtained from the disparity map, reprojected in the kinect camera

    Mat kinectCamMatrix; // Camera matrix
    Mat kinectDistortion;
    Mat perspTransform; // Perspective transformation matrix
    Mat R; // Rotation matrix from the camera of the stereo system to the camera of the kinect
    Vec3d rotVec; // Rotation vector from the camera of the stereo system to the camera of the kinect
    Vec3d T; // translation vector from the camera of the stereo system to the camera of the kinect
//    Matrix<float, 4, 4> stereo2kinect;
//    Matrix<float, 4, 4> kinect2stereo;
    Mat_<float> stereo2kinect;
    Mat_<float> kinect2stereo;

    int nbCorrespondences;
    std::vector<std::vector<Point3f>> errors;
    std::vector<float> meanDistErrors;
    std::vector<float> meanXErrors;
    std::vector<float> meanYErrors;
    std::vector<float> meanZErrors;
    std::vector<float> sigmaDistErrors;
    std::vector<float> sigmaXErrors;
    std::vector<float> sigmaYErrors;
    std::vector<float> sigmaZErrors;
    float meanDistError;
    float meanXError;
    float meanYError;
    float meanZError;
    float sigmaDistError;
    float sigmaXError;
    float sigmaYError;
    float sigmaZError;

public:
    PerformanceEvaluator(Mat rawDepth, Mat disparities, std::vector<Point> consideredDisparities, Mat camMatrix,
                             Mat distortion, Mat perspTransform, Mat rotation, Vec3d translation, int xOffset, int yOffset);

private:
    Point3f depthToWorld(int x, int y, int depthValue);

    float rawDepthToMilliMeters(int depthValue);

    void generate3DpointsFromRawDepth();

    void generate3DpointsFromDisparities();

    void transform3Dpoints(const vector<Point3f> input, vector<Point3f> &output, Mat_<float> M);

    void calculateErrors();

    Point3f getKinect3DPoint(int x, int y);

};


#endif //PROJECT_PERFORMANCEEVALUATOR_H
