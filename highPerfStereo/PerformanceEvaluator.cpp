//
// Created by dallas on 13.05.17.
//

#include "PerformanceEvaluator.h"
#include <fstream>
#include <Eigen/Dense>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/bind/placeholders.hpp>

using namespace Eigen;
using namespace boost;
using namespace boost::accumulators;

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

    xErrors = std::vector<std::vector<float>>(2048);
    yErrors = std::vector<std::vector<float>>(2048);
    zErrors = std::vector<std::vector<float>>(2048);
    distErrors = std::vector<std::vector<float>>(2048);
    distErrorMeans = std::vector<float>(2048);
    xErrorMeans = std::vector<float>(2048);
    yErrorMeans = std::vector<float>(2048);
    zErrorMeans = std::vector<float>(2048);
    distErrorSigmas = std::vector<float>(2048);
    xErrorSigmas = std::vector<float>(2048);
    yErrorSigmas = std::vector<float>(2048);
    zErrorSigmas = std::vector<float>(2048);

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
    transform3Dpoints(stereoPointCloud, stereoPointsInKinectFrame, stereo2kinect);
    std::ofstream outputFile4("stereoPointCloud_transformed.txt");
    for (it = stereoPointsInKinectFrame.begin(); it < stereoPointsInKinectFrame.end(); it++) {
        if (it->z > 0 && sqrt(pow(it->x,2)+pow(it->y,2)+pow(it->z,2)) < 8000)
            outputFile4 << it->x << " " <<  it->y << " " << it->z << " " << 0 << " " << 0 << " " << 255 << std::endl;
    }
    outputFile4.close();

    // Generate the reprojection of the 3D points obtained from the disparity map, through the kinect camera
    projectPoints(stereoPointCloud, rotVec, T, kinectCamMatrix, kinectDistortion, stereoPointsInKinectImage);

//    std::cout << stereoPointCloud.size() << " - " << stereoPointsInKinectImage.size() << std::endl;

//    // Temporary: generate reprojected pixels file
//    std::ofstream outputFile2("reprojected_points.txt");
//    std::vector<Point2f>::iterator it;
//    for (it = stereoPointsInKinectImage.begin(); it < stereoPointsInKinectImage.end(); it++) {
//        outputFile2 << it->x << " " <<  it->y << std::endl;
//    }
//    outputFile2.close();


    calculateErrors();
    std::cout << "mean dist error = " << meanDistError << std::endl;
    std::cout << "sigma dist error = " << sigmaDistError << std::endl;
    std::cout << "mean dist rel error = " << meanRelDistError << std::endl;
    std::cout << "sigma dist rel error = " << sigmaRelDistError << std::endl;
    std::cout << "nb corresp = " << nbCorrespondences << std::endl;
    std::cout << "total nb = " << stereoPointCloud.size() << std::endl;

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
//        std::cout << newP.x << " " << newP.y << " " << newP.z << std::endl;
        output.push_back(newP);
    }
}

Point3f PerformanceEvaluator::getKinect3DPoint(int x, int y) {
    return kinectPointCloud.at(x*kinectRawDepth.rows+y);
}

void PerformanceEvaluator::calculateErrors() {

    cv::Rect ROI(0,0,640,480);
    nbCorrespondences = 0;

    std::vector<Point3f>::const_iterator cloudIter;
    std::vector<Point2f>::const_iterator reprojectionIter;

    std::ofstream kinectPointsFile("kinect_matched_points.txt");
    std::ofstream stereoPointsFile("stereo_matched_points.txt");

    for(cloudIter = stereoPointsInKinectFrame.begin(), reprojectionIter = stereoPointsInKinectImage.begin();
        cloudIter < stereoPointsInKinectFrame.end();
        cloudIter++, reprojectionIter++) {

        Point3f ps = *cloudIter;
        Point2f pixel = *reprojectionIter;

        if(ROI.contains(pixel) && ps.z > 0 && ps.z <= 5000) {

            Point3f pk = getKinect3DPoint(pixel.x, pixel.y);

            if(pk.z > 0) {
                ++nbCorrespondences;

                float xError = ps.x - pk.x;
                float yError = ps.y - pk.y;
                float zError = ps.z - pk.z;
                float distError = sqrt(pow(xError,2)+pow(yError,2)+pow(zError,2));

                xErrors.at(kinectRawDepth.at<int>(pixel)).push_back(xError);
                yErrors.at(kinectRawDepth.at<int>(pixel)).push_back(yError);
                zErrors.at(kinectRawDepth.at<int>(pixel)).push_back(zError);
                distErrors.at(kinectRawDepth.at<int>(pixel)).push_back(distError);

                kinectPointsFile << pk.x << " " <<  pk.y << " " << pk.z << " " << 255 << " " << 0 << " " << 255 << std::endl;
                stereoPointsFile << ps.x << " " <<  ps.y << " " << ps.z << " " << 0 << " " << 255 << " " << 255 << std::endl;
            }
        }
    }

    kinectPointsFile.close();
    stereoPointsFile.close();

//    std::vector<std::vector<float>>::const_iterator xErrorIter;
//    std::vector<std::vector<float>>::const_iterator yErrorIter;
//    std::vector<std::vector<float>>::const_iterator zErrorIter;
//    std::vector<std::vector<float>>::const_iterator distErrorIter;
//    for (xErrorIter = xErrors.begin(), yErrorIter = yErrors.begin(), zErrorIter = zErrors.begin(), distErrorIter = distErrors.begin();
//         xErrorIter < xErrors.end();
//         xErrorIter++, yErrorIter++, zErrorIter++, distErrorIter++) {

    accumulator_set<float, stats<tag::variance(lazy)>> total_x_acc;
    accumulator_set<float, stats<tag::variance(lazy)>> total_y_acc;
    accumulator_set<float, stats<tag::variance(lazy)>> total_z_acc;
    accumulator_set<float, stats<tag::variance(lazy)>> total_dist_acc;

    for (unsigned long i = 0; i < xErrors.size(); ++i) {

        std::vector<float>* x_vec = &xErrors.at(i);
        accumulator_set<float, stats<tag::variance(lazy)>> x_acc;
        x_acc = std::for_each(x_vec->begin(), x_vec->end(), x_acc);
        total_x_acc = std::for_each(x_vec->begin(), x_vec->end(), total_x_acc);
        xErrorMeans.at(i) = boost::accumulators::extract::mean(x_acc);
        xErrorSigmas.at(i) = sqrt(boost::accumulators::variance(x_acc));

        std::vector<float>* y_vec = &yErrors.at(i);
        accumulator_set<float, stats<tag::variance(lazy)>> y_acc;
        y_acc = std::for_each(y_vec->begin(), y_vec->end(), y_acc);
        total_y_acc = std::for_each(y_vec->begin(), y_vec->end(), total_y_acc);
        yErrorMeans.at(i) = boost::accumulators::extract::mean(y_acc);
        yErrorSigmas.at(i) = sqrt(variance(y_acc));

        std::vector<float>* z_vec = &zErrors.at(i);
        accumulator_set<float, stats<tag::variance(lazy)>> z_acc;
        z_acc = std::for_each(z_vec->begin(), z_vec->end(), z_acc);
        total_z_acc = std::for_each(z_vec->begin(), z_vec->end(), total_z_acc);
        zErrorMeans.at(i) = boost::accumulators::extract::mean(z_acc);
        zErrorSigmas.at(i) = sqrt(variance(z_acc));

        std::vector<float>* dist_vec = &distErrors.at(i);
        accumulator_set<float, stats<tag::variance(lazy)>> dist_acc;
        dist_acc = std::for_each(dist_vec->begin(), dist_vec->end(), dist_acc);
        total_dist_acc = std::for_each(dist_vec->begin(), dist_vec->end(), total_dist_acc);
        distErrorMeans.at(i) = boost::accumulators::extract::mean(dist_acc);
        distErrorSigmas.at(i) = sqrt(variance(dist_acc));
        if (i != 2047 && zErrors.at(i).size() > 0) {
            relDistErrorMeans.push_back(distErrorMeans.at(i) / rawDepthToMilliMeters(i));
        }

    }

    meanDistError = boost::accumulators::extract::mean(total_dist_acc);
    meanXError = boost::accumulators::extract::mean(total_x_acc);
    meanYError = boost::accumulators::extract::mean(total_y_acc);
    meanZError = boost::accumulators::extract::mean(total_z_acc);
    sigmaDistError = sqrt(variance(total_dist_acc));
    sigmaXError = sqrt(variance(total_x_acc));
    sigmaYError = sqrt(variance(total_y_acc));
    sigmaZError = sqrt(variance(total_z_acc));

    accumulator_set<float, stats<tag::variance(lazy)>> dist_rel_acc;
    dist_rel_acc = std::for_each(relDistErrorMeans.begin(), relDistErrorMeans.end(), dist_rel_acc);
    meanRelDistError = boost::accumulators::extract::mean(dist_rel_acc);
    sigmaRelDistError = sqrt(variance(dist_rel_acc));

//    accumulator_set<float, stats<tag::variance(lazy)>> mean_x_acc;
//    std::for_each(xErrorMeans.begin(), xErrorMeans.end(), bind<void>(ref(mean_x_acc), _1));
//    meanXError = mean(mean_x_acc);
//
//    accumulator_set<float, stats<tag::variance(lazy)>> mean_y_acc;
//    std::for_each(yErrorMeans.begin(), yErrorMeans.end(), bind<void>(ref(mean_y_acc), _1));
//    meanYError = mean(mean_y_acc);
//
//    accumulator_set<float, stats<tag::variance(lazy)>> mean_z_acc;
//    std::for_each(zErrorMeans.begin(), zErrorMeans.end(), bind<void>(ref(mean_z_acc), _1));
//    meanZError = mean(mean_z_acc);
//
//    accumulator_set<float, stats<tag::variance(lazy)>> mean_dist_acc;
//    std::for_each(distErrorMeans.begin(), distErrorMeans.end(), bind<void>(ref(mean_dist_acc), _1));
//    meanDistError = mean(mean_dist_acc);

}
