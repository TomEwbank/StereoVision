/*
 * Author: Konstantin Schauwecker
 * Year:   2012
 */

// This is a minimalistic example-sparsestereo on how to use the extended
// FAST feature detector and the sparse stereo matcher.

#include <opencv2/opencv.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <vector>
#include <iostream>
#include <sparsestereo/exception.h>
#include <sparsestereo/extendedfast.h>
#include <sparsestereo/stereorectification.h>
#include <sparsestereo/sparsestereo-inl.h>
#include <sparsestereo/census-inl.h>
#include <sparsestereo/imageconversion.h>
#include <sparsestereo/censuswindow.h>
#include <fade2d/Fade_2D.h>
#include <unordered_map>
#include <plane/Plane.h>
#include <math.h>
#include <algorithm>
#include <exception>
//#include <pcl/impl/point_types.hpp>
//#include <pcl/common/projection_matrix.h>
//#include <pcl/visualization/cloud_viewer.h>
#include "highPerfStereoLib.h"
#include "GroundThruth.h"
#include "PerformanceEvaluator.h"

using namespace std;
using namespace cv;
using namespace sparsestereo;
using namespace boost;
using namespace boost::posix_time;
using namespace GEOM_FADE2D;


int main(int argc, char** argv) {
    try {

        // Stereo matching parameters
        double uniqueness = 0.5;
        int maxDisp = 70;
        int leftRightStep = 1;
        int costAggrWindowSize = 11;
        uchar gradThreshold = 50;//25; // [0,255], disparity will be computed only for points with a higher absolute gradient
        char tLow = 3;
        char tHigh = 15;
        int nIters = 3;
        double resizeFactor = 1;
        bool applyBlur = true;
        bool applyHistEqualization = true;
        int blurSize = 5;

        // Feature detection parameters
        double adaptivity = 0.25;
        int minThreshold = 3;
        bool traceLines = false;
        int lineNb = 10;
        int lineSize = 1;

        // Misc. parameters
        bool recordFullDisp = true;
        bool showImages = true;


//        // Parse arguments
//        if(argc != 3 && argc != 4) {
//            cout << "Usage: " << argv[0] << " LEFT-IMG RIGHT-IMG [CALIBRARION-FILE]" << endl;
//            return 1;
//        }
//        char* leftFile = argv[1];
//        char* rightFile = argv[2];
//        char* calibFile = argc == 4 ? argv[3] : NULL;

        String folderName = "kinect_test_imgs/"; //"test_imgs/";
        String pairName = "groundtruth";//"1_500_02";
        String imNumber = "6";
        String leftFile = folderName + "left_" + pairName + "_" + imNumber + "_rectified.ppm";
        String rightFile = folderName + "right_" + pairName + "_" + imNumber + "_rectified.ppm";
        String calibFile = folderName+ "stereoCalib_1305.yml";//"stereoMatlabCalib.yml";
        String kinectCalibFile = folderName + "kinectCalib_1305.yml";
        String rawDepthFile = folderName + "rawDepth_" + pairName + "_" + imNumber + ".yml";
        String groundTruthFile = folderName + "dist_" + pairName;

        std::vector<double> timeProfile;


        // Read input images
        cv::Mat_<unsigned char> leftImgInit, rightImgInit, colorLeftImg;
        leftImgInit = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImgInit = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);
        colorLeftImg = imread(leftFile, CV_LOAD_IMAGE_COLOR);

        if(leftImgInit.data == NULL || rightImgInit.data == NULL)
            throw sparsestereo::Exception("Unable to open input images!");

        FileStorage fs;
        fs.open(calibFile, FileStorage::READ);
        Rect commonROI;
        fs["common_ROI"] >> commonROI;
        leftImgInit = leftImgInit(commonROI);
        rightImgInit = rightImgInit(commonROI);

        cv::Mat_<unsigned char> leftImg, rightImg;
        resize(leftImgInit, leftImg, Size(), resizeFactor, resizeFactor);
        resize(rightImgInit, rightImg, Size(), resizeFactor, resizeFactor);

        // Crop image so that SSE implementation won't crash
        //cv::Rect myROI(0,0,1232,1104);
        cv::Rect myROI(0,0,16*(leftImg.cols/16),16*(leftImg.rows/16));
        leftImg = leftImg(myROI);
        rightImg = rightImg(myROI);

        //equalizeHist(leftImg, leftImg);
        //equalizeHist(rightImg, rightImg);

        // Apply Laplace function
        Mat grd, abs_grd;
        int kernel_size = 3;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;

        ptime lastTime = microsec_clock::local_time();
        cv::Laplacian( leftImg, grd, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grd, abs_grd );
        time_duration elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for gradient calculation: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;
        timeProfile.push_back(elapsed.total_microseconds()/1.0e6);

        if (showImages) {
            // Show what you got
            namedWindow("Gradient left");
            imshow("Gradient left", abs_grd);
            waitKey(0);
        }

        // Init disparity map
        Mat_<float> disparities(leftImg.rows, leftImg.cols, (float) 0);

        lastTime = microsec_clock::local_time();
        // Get the set of high gradient points
        vector<Point> highGradPoints;
        int v = 0;
        Mat highGradMask(grd.rows, grd.cols, CV_8U, Scalar(0));
        for(int j = 2; j < abs_grd.rows-2; ++j) {
            uchar* pixel = abs_grd.ptr(j);
            for (int i = 2; i < abs_grd.cols-2; ++i) {
                if (pixel[i] > gradThreshold) {
                    highGradPoints.push_back(Point(i, j));
                    highGradMask.at<uchar>(highGradPoints[v]) = (uchar) 200;
                    ++v;
                }
            }
        }
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time to get high gradient pixel set: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;
        timeProfile.push_back(elapsed.total_microseconds()/1.0e6);

        if (showImages) {
            // Show what you got
            namedWindow("Gradient mask");
            imshow("Gradient mask", highGradMask);
            waitKey(0);
        }


        if (applyHistEqualization) {
            lastTime = microsec_clock::local_time();

            equalizeHist(leftImg, leftImg);
            equalizeHist(rightImg, rightImg);

            elapsed = (microsec_clock::local_time() - lastTime);
            cout << "Time to equalize hist: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;
            timeProfile.push_back(elapsed.total_microseconds()/1.0e6);
        }

        if (applyBlur) {
            lastTime = microsec_clock::local_time();

            GaussianBlur(leftImg, leftImg, Size(blurSize, blurSize), 0, 0);
            GaussianBlur(rightImg, rightImg, Size(blurSize, blurSize), 0, 0);

            elapsed = (microsec_clock::local_time() - lastTime);
            cout << "Time to apply gaussian blur: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;
            timeProfile.push_back(elapsed.total_microseconds()/1.0e6);
        }


        // Horizontal lines tracing in images for better feature detection
        lastTime = microsec_clock::local_time();
        cv::Mat_<unsigned char> leftImgAltered, rightImgAltered;
        leftImgAltered = leftImg.clone();
        rightImgAltered = rightImg.clone();
        if (traceLines) {
            int stepSize = leftImgAltered.rows / lineNb;
            for (int k = stepSize / 2; k < leftImgAltered.rows; k = k + stepSize) {
                for (int j = k; j < k + lineSize && j < leftImgAltered.rows; ++j) {
                    leftImgAltered.row(j).setTo(0);
                    rightImgAltered.row(j).setTo(0);
                }
            }
            elapsed = (microsec_clock::local_time() - lastTime);
            cout << "Time for line drawing: " << elapsed.total_microseconds() / 1.0e6 << "s" << endl;
            timeProfile.push_back(elapsed.total_microseconds() / 1.0e6);
        }


        if (showImages) {
            // Show what you got
            namedWindow("left altered image");
            imshow("left altered image", leftImgAltered);
            waitKey(0);
            namedWindow("right altered image");
            imshow("right altered image", rightImgAltered);
            waitKey(0);
        }

        // Load rectification data
        StereoRectification* rectification = NULL;
//        if(calibFile != NULL)
//            rectification = new StereoRectification(CalibrationResult(calibFile));

        // The stereo matcher. SSE Optimized implementation is only available for a 5x5 window
        SparseStereo<CensusWindow<5>, short> stereo(maxDisp, 1, uniqueness,
                                                    rectification, false, false, leftRightStep);

        // Feature detectors for left and right image
        FeatureDetector* leftFeatureDetector = new ExtendedFAST(true, minThreshold, adaptivity, false, 2);
        FeatureDetector* rightFeatureDetector = new ExtendedFAST(false, minThreshold, adaptivity, false, 2);

        lastTime = microsec_clock::local_time();
        vector<SparseMatch> correspondences;

        // Objects for storing final and intermediate results
        cv::Mat_<char> charLeft(leftImg.rows, leftImg.cols),
                charRight(rightImg.rows, rightImg.cols);
        Mat_<unsigned int> censusLeft(leftImg.rows, leftImg.cols),
                censusRight(rightImg.rows, rightImg.cols);
        vector<KeyPoint> keypointsLeft, keypointsRight;

        // Featuredetection. This part can be parallelized with OMP
#pragma omp parallel sections default(shared) num_threads(2)
        {
#pragma omp section
            {
                keypointsLeft.clear();
                leftFeatureDetector->detect(leftImgAltered, keypointsLeft);
                ImageConversion::unsignedToSigned(leftImg, &charLeft);
                Census::transform5x5(charLeft, &censusLeft);
            }
#pragma omp section
            {
                keypointsRight.clear();
                rightFeatureDetector->detect(rightImgAltered, keypointsRight);
                ImageConversion::unsignedToSigned(rightImg, &charRight);
                Census::transform5x5(charRight, &censusRight);
            }
        }

        // Stereo matching. Not parallelized (overhead too large)
        stereo.match(censusLeft, censusRight, keypointsLeft, keypointsRight, &correspondences);


        // Print statistics
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for stereo matching: " << elapsed.total_microseconds()/1.0e6 << "s" << endl
             << "Features detected in left image: " << keypointsLeft.size() << endl
             << "Features detected in right image: " << keypointsRight.size() << endl
             << "Percentage of matched features: " << (100.0 * correspondences.size() / keypointsLeft.size()) << "%" << endl;
        timeProfile.push_back(elapsed.total_microseconds() / 1.0e6);

        if (showImages) {

            // Highlight matches as colored boxes
            Mat_<Vec3b> screen(leftImg.rows, leftImg.cols);
            cvtColor(leftImg, screen, CV_GRAY2BGR);
//        cvtColor(screen, screen, CV_BGR2HLS);
//        namedWindow("BGR2HLS");
//        imshow("BGR2HLS", screen);
//        waitKey();

            for (int i = 0; i < (int) correspondences.size(); i++) {
                double scaledDisp = (double) correspondences[i].disparity() / maxDisp;
                Vec3b color = HLS2BGR(scaledDisp * 359, 0.5, 1);
//            cout << "HLS returned = " << (int) color.val[0] << "," << (int) color.val[1] << "," << (int) color.val[2] << endl;
//            color = ConvertColor(color, CV_HLS2BGR);
//            cout << "RGB = " << (int) color.val[0] << "," << (int) color.val[1] << "," << (int) color.val[2] << endl;
//            if(scaledDisp > 0.5)
//                color = Vec3b(0, (1 - scaledDisp)*512, 255);
//            else color = Vec3b(0, 255, scaledDisp*512);

                rectangle(screen, correspondences[i].imgLeft->pt - Point2f(2, 2),
                          correspondences[i].imgLeft->pt + Point2f(2, 2),
                          (Scalar) color, CV_FILLED);
            }


            // Display image and wait
            namedWindow("Sparse stereo");
            imshow("Sparse stereo", screen);
            waitKey();
        }


        // Create the triangulation mesh & the color disparity map
        Fade_2D dt;
        lastTime = microsec_clock::local_time();
        for(int i=0; i<(int)correspondences.size(); i++) {
            float x = correspondences[i].imgLeft->pt.x;
            float y = correspondences[i].imgLeft->pt.y;
            float d = correspondences[i].disparity();

            disparities.at<float>(Point(x,y)) = d;

            Point2 p(x, y);
            dt.insert(p);
        }

        // Init final disparity map and cost map
        Mat_<float> finalDisp(leftImg.rows, leftImg.cols, (float) 0);
        Mat_<char> finalCosts(leftImg.rows, leftImg.cols, (char) 25);
        unsigned int occGridSize = 64;

        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for mesh + disp + init final disp & costs: " << elapsed.total_microseconds() / 1.0e6 << "s" << endl;
        timeProfile.push_back(elapsed.total_microseconds() / 1.0e6);

//        for(int j = 2; j<leftImg.rows-2; ++j) {
//            float* fdisp = finalDisp.ptr<float>(j);
//
//            for(int i=2; i<leftImg.cols-2; ++i) {
//                InvalidMatch p = {i,j,0};
//                ConfidentSupport cs = epipolarMatching(censusLeft, censusRight, p, maxDisp);
//                //cout  << "<< " << cs.x << ", " << cs.y << ", " << cs.disparity << ", " << cs.cost << endl;
//                fdisp[i] = cs.disparity;
//            }
//        }



        for (int iter = 1; iter <= nIters; ++iter) {

            // Fill lookup table for plane parameters
            lastTime = microsec_clock::local_time();
            unordered_map<MeshTriangle, Plane> planeTable;
            std::set<std::pair<Point2 *, Point2 *> > sEdges;
            std::vector<Triangle2 *> vAllTriangles;
            dt.getTrianglePointers(vAllTriangles);
            for (std::vector<Triangle2 *>::iterator it = vAllTriangles.begin();
                 it != vAllTriangles.end(); ++it) {

                MeshTriangle mt = {*it};
                unordered_map<MeshTriangle, Plane>::const_iterator got = planeTable.find(mt);
                if (got == planeTable.end()) {
                    Plane plane = Plane(*it, disparities);
                    planeTable[mt] = plane;
                }
                // TODO optimize by not recompute plane parameters for unchanged triangles

                if (showImages) {
                    // Use this loop over the triangles to retreive all unique edges to display
                    for (int i = 0; i < 3; ++i) {
                        Point2 *p0((*it)->getCorner((i + 1) % 3));
                        Point2 *p1((*it)->getCorner((i + 2) % 3));
                        if (p0 > p1) std::swap(p0, p1);
                        sEdges.insert(std::make_pair(p0, p1));
                    }
                }
            }
            elapsed = (microsec_clock::local_time() - lastTime);
            cout << "Time to build plane lookup table: " << elapsed.total_microseconds() / 1.0e6 << "s" << endl;
            timeProfile.push_back(elapsed.total_microseconds() / 1.0e6);

            if (showImages) {
                // Display mesh
                Mat_<Vec3b> mesh(leftImg.rows, leftImg.cols);
                cvtColor(leftImg, mesh, CV_GRAY2BGR);
                set<std::pair<Point2 *, Point2 *>>::const_iterator pos;

                for (pos = sEdges.begin(); pos != sEdges.end(); ++pos) {

                    Point2 *p1 = pos->first;
                    float scaledDisp = disparities.at<float>(Point(p1->x(), p1->y())) / maxDisp;
                    Vec3b color1;
                    if (scaledDisp > 0.5)
                        color1 = Vec3b(0, (1 - scaledDisp) * 512, 255);
                    else color1 = Vec3b(0, 255, scaledDisp * 512);

                    Point2 *p2 = pos->second;
                    scaledDisp = disparities.at<float>(Point(p2->x(), p2->y())) / maxDisp;
                    Vec3b color2;
                    if (scaledDisp > 0.5)
                        color2 = Vec3b(0, (1 - scaledDisp) * 512, 255);
                    else color2 = Vec3b(0, 255, scaledDisp * 512);


                    line2(mesh, Point(p1->x(), p1->y()), Point(p2->x(), p2->y()), (Scalar) color1, (Scalar) color2);
                }


                // Display image and wait
                namedWindow("Triangular mesh");
                imshow("Triangular mesh", mesh);
                waitKey();
            }


            // Disparity interpolation
            lastTime = microsec_clock::local_time();
            for (int j = 0; j < disparities.rows; ++j) {
                float *pixel = disparities.ptr<float>(j);
                for (int i = 0; i < disparities.cols; ++i) {
                    Point2 pointInPlaneFade = Point2(i,j);
                    Triangle2 *t = dt.locate(pointInPlaneFade);
                    MeshTriangle mt = {t};

                    if (t != NULL) {
                        unordered_map<MeshTriangle, Plane>::const_iterator got = planeTable.find(mt);
                        Plane plane;
                        if (got == planeTable.end()) {
                            plane = Plane(t, disparities);
                            planeTable[mt] = plane;
                        } else {
                            plane = got->second;
                        }
                        pixel[i] = plane.getDepth(pointInPlaneFade);
                    }

                }
            }
            elapsed = (microsec_clock::local_time() - lastTime);
            cout << "Time to build dense dipsarity map: " << elapsed.total_microseconds() / 1.0e6 << "s" << endl;
            cout << "plane table size: " << planeTable.size() << endl;


            // Display interpolated disparities
            cv::Mat dst = disparities / maxDisp;

            if (showImages) {
                namedWindow("Full interpolated disparities");
                imshow("Full interpolated disparities", dst);
                waitKey();
            }

            if (recordFullDisp) {
                Mat outputImg;
                Mat temp = dst * 255;
                temp.convertTo(outputImg, CV_8UC1);
                imwrite("disparity" + to_string(iter) + ".png", outputImg);
            }

            lastTime = microsec_clock::local_time();
            Mat_<char> matchingCosts(leftImg.rows, leftImg.cols, tHigh);
            costEvaluation(censusLeft, censusRight, highGradPoints, disparities, matchingCosts);
            PotentialSupports ps = disparityRefinement(highGradPoints, disparities, matchingCosts,
                                                        tLow, tHigh, occGridSize, finalDisp, finalCosts);
            elapsed = (microsec_clock::local_time() - lastTime);
            cout << "Time for cost eval. + disp. refinement: " << elapsed.total_microseconds() / 1.0e6 << "s" << endl;
            timeProfile.push_back(elapsed.total_microseconds() / 1.0e6);

            if(showImages) {

                // Highlight matches as colored boxes
                Mat_<Vec3b> badPts(leftImg.rows, leftImg.cols);
                cvtColor(leftImg, badPts, CV_GRAY2BGR);

                for (unsigned int i = 0; i < ps.getOccGridWidth(); ++i) {
                    for (unsigned int j = 0; j < ps.getOccGridHeight(); ++j) {
                        InvalidMatch p = ps.getInvalidMatch(i, j);
                        rectangle(badPts, Point2f(p.x, p.y) - Point2f(2, 2),
                                  Point2f(p.x, p.y) + Point2f(2, 2),
                                  (Scalar) Vec3b(0, 255, 0), CV_FILLED);
                    }
                }

                namedWindow("Candidates for epipolar matching");
                imshow("Candidates for epipolar matching", badPts);
                waitKey();


                // Display interpolated disparities for high gradient points
                //cv::normalize(finalDisp, dst, 0, 1, cv::NORM_MINMAX);
                dst = finalDisp / maxDisp;
//            namedWindow("High gradient disparities");
//            imshow("High gradient disparities", dst);
//            waitKey();

                Mat finalColorDisp(finalDisp.rows, finalDisp.cols, CV_8UC3, Scalar(0, 0, 0));
                for (int y = 0; y < finalColorDisp.rows; ++y) {
                    Vec3b *colorPixel = finalColorDisp.ptr<Vec3b>(y);
                    float *pixel = dst.ptr<float>(y);
                    for (int x = 0; x < finalColorDisp.cols; ++x)
                        if (pixel[x] > 0) {
                            Vec3b color;
                            if (pixel[x] > 0.5)
                                color = Vec3b(0, (1 - pixel[x]) * 512, 255);
                            else color = Vec3b(0, 255, pixel[x] * 512);
                            colorPixel[x] = color;
                        }
                }


                namedWindow("High gradient color disparities");
                imshow("High gradient color disparities", finalColorDisp);
                waitKey();
            }

            if (iter != nIters) {

                // Support resampling
                lastTime = microsec_clock::local_time();
                supportResampling(dt, ps, censusLeft, censusRight, 5, costAggrWindowSize, disparities, tLow, tHigh, maxDisp);
                occGridSize = max((unsigned int) 1, occGridSize / 2);
                elapsed = (microsec_clock::local_time() - lastTime);
                cout << "Time for support resampling: " << elapsed.total_microseconds() / 1.0e6 << "s" << endl;
                timeProfile.push_back(elapsed.total_microseconds() / 1.0e6);
            }
        }

        double totalTime = 0;
        for(vector<double>::iterator it = timeProfile.begin() ; it < timeProfile.end(); it++) {
            totalTime += *it;
        }
        cout << "Total time: " << totalTime << endl << "Fps: " << 1/totalTime << endl;

/*------------------------------------------------------------------------------------*/


        Mat Q;
        fs["Q"] >> Q;

/*        // True distance error estimation

        ifstream readFile(groundTruthFile);
        vector<GroundThruth> groundTruthVec;
        GroundThruth data;
        while(readFile >> data) {
            //cout << data.x << ", " << data.y << ", " << data.disparity << ", " << data.distance << ", " << data.pointName << endl;
            groundTruthVec.push_back(data);
        }

//        for (int l = 0; l < groundTruthVec.size(); ++l) {
//            data = groundTruthVec[l];
//            cout << data.x << ", " << data.y << ", " << data.disparity << ", " << data.distance << ", " << data.pointName << endl;
//        }

        std::vector<Vec3d> vin(groundTruthVec.size());
        std::vector<Vec3d> vout(groundTruthVec.size());
        cout << "Truth points nb: " << groundTruthVec.size() << endl;
        double meanDispError = 0;
        int nbSkipped = 0;
        for (int n = 0; n < groundTruthVec.size(); ++n) {
            GroundThruth t = groundTruthVec[n];
            Point2i coordInROI = t.getCoordInROI(commonROI);
            float disp = disparities.at<float>(coordInROI);
            double dispError = abs(disp-t.disparity);
            if (dispError < 15) {
                meanDispError += dispError;
            } else {
                ++nbSkipped;
            }
            Vec3d p(t.x, t.y, disparities.at<float>(coordInROI));
            vin[n] = p;
        }
        cout << nbSkipped  << endl;
        meanDispError = meanDispError/(groundTruthVec.size()-nbSkipped);
        cout << "Mean disparity error: " << meanDispError << endl;

        perspectiveTransform(vin, vout, Q);

        double meanError = 0;
        nbSkipped = 0;
        ofstream smallCloudFile("smallCloud.txt");
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_xyzrgb (new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int i = 0; i < groundTruthVec.size(); ++i) {
            Vec3d point3D = vout[i];
            double x = point3D.val[0];
            double y = point3D.val[1];
            double z = point3D.val[2];
            double b = colorLeftImg.at<Vec3b>(Point(vin[i].val[0], vin[i].val[1])).val[0];
            double g = colorLeftImg.at<Vec3b>(Point(vin[i].val[0], vin[i].val[1])).val[1];
            double r = colorLeftImg.at<Vec3b>(Point(vin[i].val[0], vin[i].val[1])).val[2];


//            pcl::PointXYZRGB pclPoint;
//            pclPoint.x = x;
//            pclPoint.y = y;
//            pclPoint.z = z;
//            pclPoint.b = colorLeftImg.at<Vec3b>(Point(vin[i].val[0], vin[i].val[1])).val[0];
//            pclPoint.g = colorLeftImg.at<Vec3b>(Point(vin[i].val[0], vin[i].val[1])).val[1];
//            pclPoint.r = colorLeftImg.at<Vec3b>(Point(vin[i].val[0], vin[i].val[1])).val[2];
//            cloud_xyzrgb->push_back(pclPoint);

            double error = abs(groundTruthVec[i].distance - sqrt(pow(x,2)+pow(y,2)+pow(z,2)));
            //cout << x << ", " << y << ", " << z << endl;
            //cout << error << endl;
            if (error < 2000) {
                meanError += error;
                smallCloudFile << x << " " << y << " " << z << " " << r << " " << g  << " " << b << endl;
            } else {
                ++nbSkipped;
            }
        }
        cout << nbSkipped  << endl;
        meanError = meanError/(groundTruthVec.size()-nbSkipped);
        cout << "Mean dist error: " << meanError << endl;
        smallCloudFile.close();

//        pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
//        viewer.showCloud(cloud_xyzrgb);

//        waitKey(0);

*/

/* --------------------------------------------------------------------------------- */

        // Generate pointCloud
        std::vector<Vec3d> vin2(highGradPoints.size());
        std::vector<Vec3d> vout2(highGradPoints.size());
        std::vector<Point>::iterator ptsIt;
        std::vector<Vec3d>::iterator vin2It;
        for (ptsIt = highGradPoints.begin(), vin2It = vin2.begin();
             ptsIt < highGradPoints.end();
             ptsIt++, vin2It++) {

            Point coordInROI = *ptsIt;
            Vec3d p(coordInROI.x+commonROI.x, coordInROI.y+commonROI.y, finalDisp.at<float>(coordInROI));
            *vin2It = p;
        }
        perspectiveTransform(vin2, vout2, Q);
        ofstream outputFile("pointCloud_"+pairName+".txt");
        std::vector<Vec3d>::iterator vout2It;
        for (ptsIt = highGradPoints.begin(), vout2It = vout2.begin(); vout2It < vout2.end(); vout2It++, ptsIt++) {
            Vec3d point3D = *vout2It;
            Point pointInImage = *ptsIt;
            pointInImage.x += commonROI.x;
            pointInImage.y += commonROI.y;

            double x = point3D.val[0];
            double y = point3D.val[1];
            double z = point3D.val[2];

            Vec3b color = colorLeftImg.at<Vec3b>(pointInImage);
            double r = color.val[2];
            double g = color.val[1];
            double b = color.val[0];

            if (z > 0 && sqrt(pow(x,2)+pow(y,2)+pow(z,2)) < 8000)
                outputFile << x << " " << y << " " << z << " " << r << " " << g  << " " << b << endl;
        }
        outputFile.close();

/* -----------------------------------------------------------------------------------*/

        // Stereo VS Kinect, error calculation

        FileStorage fsKinect;
        fsKinect.open(kinectCalibFile, FileStorage::READ);
        Mat kinectCamMatrix;
        fsKinect["K2"] >> kinectCamMatrix;
        Mat kinectDistortion;
        fsKinect["D2"] >> kinectDistortion;
        Mat R;
        fsKinect["R"] >> R;
        Vec3d T;
        fsKinect["T"] >> T;

        FileStorage fsRawDepth;
        fsRawDepth.open(rawDepthFile, FileStorage::READ);
        Mat rawDepth;
        fsRawDepth["rawDepth"] >> rawDepth;

        PerformanceEvaluator evaluator(rawDepth, finalDisp, highGradPoints, kinectCamMatrix, kinectDistortion, Q, R, T,
                                       commonROI.x, commonROI.y);



/* ---------------------------------------------------------------------------------------*/

        // Clean up
        delete leftFeatureDetector;
        delete rightFeatureDetector;
        if(rectification != NULL)
            delete rectification;

//        //tests
//        HammingDistance h;
//        unsigned int a = 15;
//        unsigned int b = 491535;
//        int d = h.calculate(b,a);
//        cout << d << endl;

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}

// TODO display point cloud -> DONE WITH MESHLAB
// TODO add second image in test -> DONE
// TODO check influence of auto contrast techniques
// TODO replace cramer plane param calculation to increase speed
// TODO fix & check the influence of census transform with variable window

