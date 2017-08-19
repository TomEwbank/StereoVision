/* +------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)            |
   |                          http://www.mrpt.org/                          |
   |                                                                        |
   | Copyright (c) 2005-2017, Individual contributors, see AUTHORS file     |
   | See: http://www.mrpt.org/Authors - All rights reserved.                |
   | Released under BSD License. See details in http://www.mrpt.org/License |
   +------------------------------------------------------------------------+ */

#include <mrpt/math/ransac.h>
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/random.h>
#include <mrpt/utils/CTicTac.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/CPointCloud.h>
#include <mrpt/opengl/stock_objects.h>
#include <mrpt/opengl/CTexturedPlane.h>

#include <opencv2/opencv.hpp>
#include <sparsestereo/exception.h>
#include <sparsestereo/extendedfast.h>
#include <sparsestereo/stereorectification.h>
#include <sparsestereo/sparsestereo-inl.h>
#include <sparsestereo/census-inl.h>
#include <sparsestereo/imageconversion.h>
#include <sparsestereo/censuswindow.h>
#include <fade2d/Fade_2D.h>
#include <unordered_map>
#include "highPerfStereoLib.h"

using namespace mrpt;
using namespace mrpt::utils;
using namespace mrpt::gui;
using namespace mrpt::math;
using namespace mrpt::random;
using namespace mrpt::poses;
using namespace std;

void ransac3Dplane_fit(
        const CMatrixDouble& allData, const vector_size_t& useIndices,
        vector<CMatrixDouble>& fitModels)
{
    ASSERT_(useIndices.size() == 3);

    TPoint3D p1(
            allData(0, useIndices[0]), allData(1, useIndices[0]),
            allData(2, useIndices[0]));
    TPoint3D p2(
            allData(0, useIndices[1]), allData(1, useIndices[1]),
            allData(2, useIndices[1]));
    TPoint3D p3(
            allData(0, useIndices[2]), allData(1, useIndices[2]),
            allData(2, useIndices[2]));

    try
    {
        TPlane plane(p1, p2, p3);
        fitModels.resize(1);
        CMatrixDouble& M = fitModels[0];

        M.setSize(1, 4);
        for (size_t i = 0; i < 4; i++) M(0, i) = plane.coefs[i];
    }
    catch (std::exception&)
    {
        fitModels.clear();
        return;
    }
}

void ransac3Dplane_distance(
        const CMatrixDouble& allData, const vector<CMatrixDouble>& testModels,
        const double distanceThreshold, unsigned int& out_bestModelIndex,
        vector_size_t& out_inlierIndices)
{
    ASSERT_(testModels.size() == 1)
    out_bestModelIndex = 0;
    const CMatrixDouble& M = testModels[0];

    ASSERT_(size(M, 1) == 1 && size(M, 2) == 4)

    TPlane plane;
    plane.coefs[0] = M(0, 0);
    plane.coefs[1] = M(0, 1);
    plane.coefs[2] = M(0, 2);
    plane.coefs[3] = M(0, 3);

    const size_t N = size(allData, 2);
    out_inlierIndices.clear();
    out_inlierIndices.reserve(100);
    for (size_t i = 0; i < N; i++)
    {
        const double d = plane.distance(
                TPoint3D(
                        allData.get_unsafe(0, i), allData.get_unsafe(1, i),
                        allData.get_unsafe(2, i)));
        if (d < distanceThreshold) out_inlierIndices.push_back(i);
    }
}

/** Return "true" if the selected points are a degenerate (invalid) case.
  */
bool ransac3Dplane_degenerate(
        const CMatrixDouble& allData, const mrpt::vector_size_t& useIndices)
{
    return false;
}

// ------------------------------------------------------
//				TestRANSAC
// ------------------------------------------------------
void TestRANSAC()
{
//    randomGenerator.randomize();
//
//    // Generate random points:
//    // ------------------------------------
//    const size_t N_plane = 300;
//    const size_t N_noise = 100;
//
//    const double PLANE_EQ[4] = {1, -1, 1, -2};
//
//    CMatrixDouble data(3, N_plane + N_noise);
//    for (size_t i = 0; i < N_plane; i++)
//    {
//        const double xx = randomGenerator.drawUniform(-3, 3);
//        const double yy = randomGenerator.drawUniform(-3, 3);
//        const double zz =
//                -(PLANE_EQ[3] + PLANE_EQ[0] * xx + PLANE_EQ[1] * yy) / PLANE_EQ[2];
//        data(0, i) = xx;
//        data(1, i) = yy;
//        data(2, i) = zz;
//    }
//
//    for (size_t i = 0; i < N_noise; i++)
//    {
//        data(0, i + N_plane) = randomGenerator.drawUniform(-4, 4);
//        data(1, i + N_plane) = randomGenerator.drawUniform(-4, 4);
//        data(2, i + N_plane) = randomGenerator.drawUniform(-4, 4);
//    }

    StereoParameters params;

    // Stereo matching parameters
    params.uniqueness = 0.35;
    params.maxDisp = 100;
    params.minDisp = 36;
    params.leftRightStep = 4;
    params.costAggrWindowSize = 11;
    params.gradThreshold = 150; // [0,255], disparity will be computed only for points with a higher absolute gradient
    params.tLow = 2;
    params.tHigh = 6;
    params.nIters = 1;
    params.resizeFactor = 1;
    params.applyBlur = true;
    params.applyHistEqualization = true;
    params.blurSize = 3;
    params.rejectionMargin = 10;
    params.occGridSize = 32;

    // Feature detection parameters
    params.adaptivity = 0.4;
    params.minThreshold = 2;
    params.traceLines = false;
    params.nbLines = 20;
    params.lineSize = 4;
    params.invertRows = false;
    params.nbRows = 20;

    // Gradient parameters
    params.kernelSize = 3;
    params.scale = 1;
    params.delta = 0;
    params.ddepth = CV_16S;

    // Misc. parameters
    params.recordFullDisp = false;
    params.showImages = false;
    params.colorMapSliding = 60;

    String folderName = "imgs_rectified/";
    String calibFile = folderName+"stereoParams_2906.yml";
    String serie = "20_50";

    FileStorage fs;
    fs.open(calibFile, FileStorage::READ);
    Rect commonROI;
    fs["common_ROI"] >> commonROI;
    Mat Q;
    fs["Q"] >> Q;

    ofstream outputFile("planeErrors_"+serie+".txt");

    String imNum[] = {"1","2","3","4","5","7","8","9","10","11"};
    double trueAlpha[] = {16.0,26.3,21.7,26.5,27.1,38.5,29.8,24.9,26.2,30.8};
    double trueBeta[] = {-2.5,-2.4,17.5,39.0,-2.2,-2.6,-1.0,-1.9,10.4,11.1};

    for(int k = 0; k<10; ++k) {
        String leftFile = "imgs_rectified/left_" + serie + "_floor_inclination_" + imNum[k] + "_rectified.png";
        String rightFile = "imgs_rectified/right_" + serie + "_floor_inclination_" + imNum[k] + "_rectified.png";

        // Read input images
        cv::Mat_<unsigned char> leftImg, rightImg;
        leftImg = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImg = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);


        Mat_<float> finalDisp(commonROI.height, commonROI.width, (float) 0);
        vector<Point> highGradPoints;

        if (leftImg.data == NULL || rightImg.data == NULL)
            throw sparsestereo::Exception("Unable to open input images!");

        // Compute disparities
        highPerfStereo(leftImg(commonROI), rightImg(commonROI), params, finalDisp, highGradPoints);

        // Generate pointCloud
        std::vector<Vec3d> vin2;
        int pass = 0;
        for (Point coordInROI : highGradPoints) {
            if(pass == 4) {
                if (coordInROI.x > 200 && coordInROI.x < 600 && coordInROI.y > 160 && coordInROI.y < 300) {
                    Vec3d p(coordInROI.x + commonROI.x, coordInROI.y + commonROI.y, finalDisp.at<float>(coordInROI));
                    vin2.push_back(p);
                }
                pass = 0;
            } else {
                ++pass;
            }
        }

        std::vector<Vec3d> vout2(vin2.size());
        perspectiveTransform(vin2, vout2, Q);
        CMatrixDouble data(3, vout2.size());
        int i = 0;
        for (Vec3d point3D : vout2) {
//        Vec3d point3D = *vout2It;
//        Point pointInImage = *ptsIt;
//        pointInImage.x += commonROI.x;
//        pointInImage.y += commonROI.y;

            data(0, i) = point3D.val[0] / 1000;
            data(1, i) = point3D.val[1] / 1000;
            data(2, i) = point3D.val[2] / 1000;
            ++i;

//        Vec3b color = colorLeftImg.at<Vec3b>(pointInImage);
//        double r = color.val[2];
//        double g = color.val[1];
//        double b = color.val[0];

//        if (z > 0 && sqrt(pow(x,2)+pow(y,2)+pow(z,2)) < 8000)
//            outputFile << x << " " << y << " " << z << " " << r << " " << g  << " " << b << endl;
        }

        // Run RANSAC
        // ------------------------------------
        CMatrixDouble best_model;
        vector_size_t best_inliers;
        const double DIST_THRESHOLD = 0.005;

        CTicTac tictac;
        const size_t TIMES = 2000;

        mrpt::math::RANSAC myransac;
//        for (size_t iters = 0; iters < TIMES; iters++)
//            myransac.execute(
//                    data, ransac3Dplane_fit, ransac3Dplane_distance,
//                    ransac3Dplane_degenerate, DIST_THRESHOLD,
//                    3,  // Minimum set of points
//                    best_inliers, best_model,
//                    iters == 0 ? mrpt::utils::LVL_DEBUG
//                               : mrpt::utils::LVL_INFO  // Verbose
//            );

        while(best_inliers.size() < 0.7*data.cols())
            myransac.execute(
                    data, ransac3Dplane_fit, ransac3Dplane_distance,
                    ransac3Dplane_degenerate, DIST_THRESHOLD,
                    3,  // Minimum set of points
                    best_inliers, best_model,
                    mrpt::utils::LVL_DEBUG  // Verbose
            );

        cout << "Computation time: " << tictac.Tac() * 1000.0 / TIMES << " ms"
             << endl;

        ASSERT_(size(best_model, 1) == 1 && size(best_model, 2) == 4)

        cout << "RANSAC finished: Best model: " << best_model << endl;
        cout << "Best inliers: " << best_inliers.size() << endl;

        TPlane plane(
                best_model(0, 0), best_model(0, 1), best_model(0, 2), best_model(0, 3));

        double PI = 3.14159265;
        double alpha = atan(best_model(0, 2) / best_model(0, 1)) * 180 / PI;
        double beta = atan(best_model(0, 0) / best_model(0, 1)) * 180 / PI;
        double alphaError = alpha - trueAlpha[k];
        double betaError = beta - trueBeta[k];

        outputFile << std::fixed << std::setprecision(2) << trueAlpha[k] << " & " << trueBeta[k] << " & " << alpha << " & " << alphaError << " & " << beta << " & " << betaError << endl;
        cout  << "true alpha = " << trueAlpha[k] << ", alpha = " << alpha << ", error =" << alphaError << endl;
        cout << "true beta = " << trueBeta[k] << ", beta = " << beta << ", error =" << betaError << endl;

        // Show GUI
        // --------------------------
        mrpt::gui::CDisplayWindow3D win("Set of points", 500, 500);
        opengl::COpenGLScene::Ptr scene = std::make_shared<opengl::COpenGLScene>();

        scene->insert(
                std::make_shared<opengl::CGridPlaneXY>(-20, 20, -20, 20, 0, 1));
        scene->insert(opengl::stock_objects::CornerXYZ());

        opengl::CPointCloud::Ptr points = std::make_shared<opengl::CPointCloud>();
        points->setColor(0, 0, 1);
        points->setPointSize(3);
        points->enableColorFromZ();

        {
            std::vector<float> xs, ys, zs;

            data.extractRow(0, xs);
            data.extractRow(1, ys);
            data.extractRow(2, zs);
            points->setAllPointsFast(xs, ys, zs);
        }

        scene->insert(points);

        opengl::CTexturedPlane::Ptr glPlane =
                std::make_shared<opengl::CTexturedPlane>(-4, 4, -4, 4);

        CPose3D glPlanePose;
        plane.getAsPose3D(glPlanePose);
        glPlane->setPose(glPlanePose);

        scene->insert(glPlane);

        win.get3DSceneAndLock() = scene;
        win.unlockAccess3DScene();
        win.forceRepaint();

        win.waitForKey();
    }
}

// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------
int main()
{
    try
    {
        TestRANSAC();
        return 0;
    }
    catch (std::exception& e)
    {
        std::cout << "MRPT exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        printf("Untyped exception!!");
        return -1;
    }
}

