//
// Created by dallas on 13.05.17.
//

#include "PerformanceEvaluator.h"
#include <fstream>
#include <Eigen/Dense>

using namespace Eigen;

PerformanceEvaluator::PerformanceEvaluator(Mat rawDepth, Mat disparities, std::vector<Point> consideredDisparities, Mat camMatrix,
                                           Mat distortion, Mat perspTransform, Mat rotation, Vec3d translation, int xOffset, int yOffset) {
    this->kinectRawDepth = rawDepth;
    this->disparities = disparities;
    this->consideredDisparities = consideredDisparities;
    this->kinectCamMatrix = camMatrix;
    this->perspTransform = perspTransform;
    this->R = rotation;
    Rodrigues(rotation, this->rotVec);
    this->T = translation;
    this->xOffset = xOffset;
    this->yOffset = yOffset;

    errors = std::vector<std::vector<Point3f>>(2048);
    meanDistErrors = std::vector<float>(2048);
    meanXErrors = std::vector<float>(2048);
    meanYErrors = std::vector<float>(2048);
    meanZErrors = std::vector<float>(2048);
    sigmaDistErrors = std::vector<float>(2048);
    sigmaXErrors = std::vector<float>(2048);
    sigmaYErrors = std::vector<float>(2048);
    sigmaZErrors = std::vector<float>(2048);

    nbCorrespondences = 0;
    meanDistError = 0;
    meanXError = 0;
    meanYError = 0;
    meanZError = 0;
    sigmaDistError = 0;
    sigmaXError = 0;
    sigmaYError = 0;
    sigmaZError = 0;

    // Create transformation matrix from stereo system frame to kinect frame and vice versa
    Mat_<float> stereo2kinect(4,4,0.0);
    R.copyTo(stereo2kinect(cv::Rect(0,0,3,3)));
    stereo2kinect.at<float>(Point(3,0)) = (float) T.val[0];
    stereo2kinect.at<float>(Point(3,1)) = (float) T.val[1];
    stereo2kinect.at<float>(Point(3,2)) = (float) T.val[2];
    stereo2kinect.at<float>(Point(3,3)) = 1.0;
    this->stereo2kinect = stereo2kinect;
    invert(stereo2kinect, this->kinect2stereo);
//    kinect2stereo.fill(0);
//    for (int i = 0; i < 3; ++i) {
//        for(int j = 0; j < 3; j++) {
//            kinect2stereo(i,j) = rotation.at<float>(j,i);
//        }
//    }
//    for (int i = 0; i < 3; ++i) {
//        kinect2stereo(i,3) = translation.val[i];
//    }
//    kinect2stereo(3,3) = 1;
//    stereo2kinect = kinect2stereo.inverse();


    // init rawdepth to meters lookup table
    for (int i = 0; i < 2048; i++) {
        depthLookUp[i] = rawDepthToMilliMeters(i);
    }

    generate3DpointsFromRawDepth();

    // Temporary: generate kinect point cloud file
    std::ofstream outputFile("kinectpointCloud.txt");
    std::vector<Point3f>::iterator it;
    for (it = kinectPointCloud.begin(); it < kinectPointCloud.end(); it++) {
        outputFile << it->x << " " <<  it->y << " " << it->z << " " << 255 << " " << 0 << " " << 0 << std::endl;
    }
    outputFile.close();

    generate3DpointsFromDisparities();

    // Temporary: generate stereo point cloud file
    std::ofstream outputFile3("stereoPointCloud.txt");
    for (it = stereoPointCloud.begin(); it < stereoPointCloud.end(); it++) {
        if (it->z > 0 && sqrt(pow(it->x,2)+pow(it->y,2)+pow(it->z,2)) < 8000)
            outputFile3 << it->x << " " <<  it->y << " " << it->z << " " << 0 << " " << 255 << " " << 0 << std::endl;
    }
    outputFile3.close();

    // Temporary: generate stereo point cloud file
    std::vector<Point3f> transformedCloud;
    transform3Dpoints(stereoPointCloud, transformedCloud, stereo2kinect);
    std::ofstream outputFile4("stereoPointCloud_transformed.txt");
    for (it = transformedCloud.begin(); it < transformedCloud.end(); it++) {
        if (it->z > 0 && sqrt(pow(it->x,2)+pow(it->y,2)+pow(it->z,2)) < 8000)
            outputFile4 << it->x << " " <<  it->y << " " << it->z << " " << 0 << " " << 0 << " " << 255 << std::endl;
    }
    outputFile4.close();

    // Generate the reprojection of the 3D points obtained from the disparity map, through the kinect camera
    projectPoints(stereoPointCloud, rotVec, T, kinectCamMatrix, kinectDistortion, stereoPointsInKinect);

    std::cout << stereoPointCloud.size() << " - " << stereoPointsInKinect.size() << std::endl;

//    // Temporary: generate reprojected pixels file
//    std::ofstream outputFile2("reprojected_points.txt");
//    std::vector<Point2f>::iterator it;
//    for (it = stereoPointsInKinect.begin(); it < stereoPointsInKinect.end(); it++) {
//        outputFile2 << it->x << " " <<  it->y << std::endl;
//    }
//    outputFile2.close();

}

void PerformanceEvaluator::generate3DpointsFromDisparities() {
    std::vector<Point3f> disparityPoints(consideredDisparities.size());
    std::vector<Point3f> pointsIn3D(consideredDisparities.size());
    std::vector<Point>::iterator ptsIt;
    std::vector<Point3f>::iterator vin2It;
    for (ptsIt = consideredDisparities.begin(), vin2It = disparityPoints.begin();
         ptsIt < consideredDisparities.end();
         ptsIt++, vin2It++) {

        Point coordInDisparityMap = *ptsIt;
        Point3f p(coordInDisparityMap.x+xOffset, coordInDisparityMap.y+yOffset, disparities.at<float>(coordInDisparityMap));
        *vin2It = p;
    }
    perspectiveTransform(disparityPoints, pointsIn3D, perspTransform);
    this->stereoPointCloud = pointsIn3D;
}

void PerformanceEvaluator::generate3DpointsFromRawDepth() {

    for (int x = 0; x < kinectRawDepth.cols; ++x) {
        for (int y = 0; y < kinectRawDepth.rows; ++y) {

            // Convert kinect data to world xyz coordinate
            Point3f p = depthToWorld(x, y, kinectRawDepth.at<int>(Point(x, y)));
            kinectPointCloud.push_back(p);
        }
    }
}

Point3f PerformanceEvaluator::depthToWorld(int x, int y, int depthValue) {
    // Kinect camera parameters
    double fx_d = 1.0 / 5.9421434211923247e+02;
    double fy_d = 1.0 / 5.9104053696870778e+02;
    double cx_d = 3.3930780975300314e+02;
    double cy_d = 2.4273913761751615e+02;

    Point3f result;
    double depth =  depthLookUp[depthValue];//rawDepthToMilliMeters(depthValue);
    result.x = (float)((x - cx_d) * depth * fx_d);
    result.y = (float)((y - cy_d) * depth * fy_d);
    result.z = (float)(depth);
    return result;
}

float PerformanceEvaluator::rawDepthToMilliMeters(int depthValue) {
    if (depthValue < 2047) {
        return (float) (1000*(1.0 / ((double)(depthValue) * -0.0030711016 + 3.3309495161)));
    }
    return 0.0f;
}

void PerformanceEvaluator::transform3Dpoints(const std::vector<Point3f> input, vector<Point3f> &output, Mat_<float> M) {

//    output.clear();
    std::vector<Point3f>::const_iterator it;
    for (it = input.begin(); it < input.end(); it++) {

        Point3f p = *it;
        Mat_<float> v(4,1,1.0);
        v.at<float>(Point(0,0)) = p.x;
        v.at<float>(Point(0,1)) = p.y;
        v.at<float>(Point(0,2)) = p.z;

        Mat_<float> newV = M*v;
        Point3f newP(newV.at<float>(Point(0,0)), newV.at<float>(Point(0,1)), newV.at<float>(Point(0,2)));
        std::cout << newP.x << " " << newP.y << " " << newP.z << std::endl;
        output.push_back(newP);
    }
}

Point3f PerformanceEvaluator::getKinect3DPoint(int x, int y) {
    return kinectPointCloud.at(x*kinectRawDepth.rows+y);
}

void PerformanceEvaluator::calculateErrors() {
    std::vector<Point3f>::const_iterator cloudIter;
    std::vector<Point2f>::const_iterator reprojectionIter;
    for(cloudIter = stereoPointCloud.begin(), reprojectionIter = stereoPointsInKinect.begin();
            cloudIter < stereoPointCloud.end();
            cloudIter++, reprojectionIter++) {

    }

}
