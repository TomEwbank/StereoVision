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
    std::vector<Point3f> stereoPointCloud; // 3D points obtained from disparity map, in the reference frame of the left stereo camera
    std::vector<Point2f> stereoPointsInKinectImage; // Pixel coordinates of the 3D points obtained from the disparity map, reprojected in the kinect camera
    std::vector<Point3f> stereoPointsInKinectFrame; // 3D points obtained from disparity map, in the reference frame of the kinect camera

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
    std::vector<std::vector<float>> xErrors;
    std::vector<std::vector<float>> yErrors;
    std::vector<std::vector<float>> zErrors;
    std::vector<std::vector<float>> distErrors;
    std::vector<float> distErrorMeans;
    std::vector<float> xErrorMeans;
    std::vector<float> yErrorMeans;
    std::vector<float> zErrorMeans;
    std::vector<float> distErrorSigmas;
    std::vector<float> xErrorSigmas;
    std::vector<float> yErrorSigmas;
    std::vector<float> zErrorSigmas;
    float meanDistError;
    float meanXError;
    float meanYError;
    float meanZError;
    float sigmaDistError;
    float sigmaXError;
    float sigmaYError;
    float sigmaZError;


    std::vector<float> relDistErrorMeans;
    float meanRelDistError;
    float sigmaRelDistError;

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
