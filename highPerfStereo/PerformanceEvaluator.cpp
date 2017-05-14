//
// Created by dallas on 13.05.17.
//

#include "PerformanceEvaluator.h"
#include <fstream>

PerformanceEvaluator::PerformanceEvaluator(Mat rawDepth, Mat disparities, std::vector<Point> consideredDisparities,
                                           Mat camMatrix, Mat perspTransform, Mat rotation, Vec3d translation) {
    this->kinectRawDepth = rawDepth;
    this->disparities = disparities;
    this->consideredDisparities = consideredDisparities;
    this->camMatrix = camMatrix;
    this->perspTransform = perspTransform;
    this->R = rotation;
    this->T = translation;

    // init rawdepth to meters lookup table
    for (int i = 0; i < 2048; i++) {
        depthLookUp[i] = rawDepthToMeters(i);
    }

    generate3DpointsFromRawDepth();

    // Temporary: generate kinect point cloud file
    std::ofstream outputFile("kinectpointCloud.txt");
    std::vector<Point3f>::iterator it;
    for (it = kinectPointCloud.begin(); it < kinectPointCloud.end(); it++) {
        std::cout << it->x << " " <<  it->y << " " << it->z << std::endl;
        outputFile << it->x << " " <<  it->y << " " << it->z << std::endl;
    }
    outputFile.close();

}

void PerformanceEvaluator::generate3DpointsFromRawDepth() {

    for (int x = 0; x < kinectRawDepth.cols; ++x) {
        for (int y = 0; y < kinectRawDepth.rows; ++y) {


            std::cout << x << std::endl;
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
    double depth =  depthLookUp[depthValue];//rawDepthToMeters(depthValue);
    result.x = (float)((x - cx_d) * depth * fx_d);
    result.y = (float)((y - cy_d) * depth * fy_d);
    result.z = (float)(depth);
    return result;
}

float PerformanceEvaluator::rawDepthToMeters(int depthValue) {
    if (depthValue < 2047) {
        return (float)(1.0 / ((double)(depthValue) * -0.0030711016 + 3.3309495161));
    }
    return 0.0f;
}
