

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
#include "GroundTruth.h"
#include "PerformanceEvaluator.h"

using namespace std;
using namespace cv;
using namespace sparsestereo;
using namespace boost;
using namespace boost::posix_time;
using namespace GEOM_FADE2D;


int main(int argc, char** argv) {
    try {

        StereoParameters params;

        // Stereo matching parameters
        params.uniqueness = 0.5;
        params.maxDisp = 70;
        params.leftRightStep = 1;
        params.costAggrWindowSize = 11;
        params.gradThreshold = 70;//25; // [0,255], disparity will be computed only for points with a higher absolute gradient
        params.tLow = 2;
        params.tHigh = 10;
        params.nIters = 1;
        params.resizeFactor = 1;
        params.applyBlur = true;
        params.applyHistEqualization = true;
        params.blurSize = 5;

        // Feature detection parameters
        params.adaptivity = 0.25;
        params.minThreshold = 3;
        params.traceLines = false;
        params.nbLines = 20;
        params.lineSize = 2;
        params.invertRows = false;
        params.nbRows = 50;

        // Gradient parameters
        params.kernelSize = 3;
        params.scale = 1;
        params.delta = 0;
        params.ddepth = CV_16S;

        // Misc. parameters
        params.recordFullDisp = false;
        params.showImages = true;
        params.colorMapSliding = 60;


        String folderName = "imgs_rectified/"; //"test_imgs/";
        String pairName = "1_200_ball_grassfloor_light";//"1_500_02";
        String imNumber = "1";
        String leftFile = folderName + "left_" + pairName + "_" + imNumber + "_rectified.png";
        String rightFile = folderName + "right_" + pairName + "_" + imNumber + "_rectified.png";
        String calibFile = folderName+ "stereoCalib_2305_rotx008_invTOnly.yml";
//        String kinectCalibFile = folderName + "kinectCalib_1305_rotsupp.yml";
//        bool kinectTransformToCorrect = true;
//        String rawDepthFile = folderName + "rawDepth_" + pairName + "_" + imNumber + ".yml";
//        String groundTruthFile = folderName + "dist_" + pairName;

        std::vector<double> timeProfile;
        ptime lastTime;
        time_duration elapsed;

        // Read input images
        cv::Mat_<unsigned char> leftImg, rightImg, colorLeftImg;
        leftImg = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImg = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);
        colorLeftImg = imread(leftFile, CV_LOAD_IMAGE_COLOR);

        Mat_<float> finalDisp(leftImg.rows, leftImg.cols, (float) 0);
        vector<Point> highGradPoints;

        if(leftImg.data == NULL || rightImg.data == NULL)
            throw sparsestereo::Exception("Unable to open input images!");

        FileStorage fs;
        fs.open(calibFile, FileStorage::READ);
        Rect commonROI;
        fs["common_ROI"] >> commonROI;

        lastTime = microsec_clock::local_time();
        highPerfStereo(leftImg, rightImg, commonROI, params, finalDisp, highGradPoints);
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time elapsed: " << elapsed.total_microseconds()/1.0e6 << endl << "Fps: " << 1/(elapsed.total_microseconds()/1.0e6) << endl;


/*------------------------------------------------------------------------------------*/


        Mat Q;
        fs["Q"] >> Q;

        // Generate pointCloud
        cout << highGradPoints.size() << endl;
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

//        // Stereo VS Kinect, error calculation
//
//        FileStorage fsKinect;
//        fsKinect.open(kinectCalibFile, FileStorage::READ);
//        Mat kinectCamMatrix;
//        fsKinect["K2"] >> kinectCamMatrix;
//        Mat kinectDistortion;
//        fsKinect["D2"] >> kinectDistortion;
//        Mat R;
//        fsKinect["R"] >> R;
//        Vec3d T;
//        fsKinect["T"] >> T;
//
//        cout << "R = " << R << endl;
//        cout << "T = " << T << endl;
//
//
//        // If manual correction needs to be applied,
//        // create new R & T that are composed of the original and an additional transformation
//        if (kinectTransformToCorrect) {
//            // Get the transformation correction to apply
//            Mat Rsupp;
//            fsKinect["Rsupp"] >> Rsupp;
//            Vec3d Tsupp;
//            fsKinect["Tsupp"] >> Tsupp;
//
//            cout << "Rsupp = " << Rsupp << endl;
//            cout << "Tsupp = " << Tsupp << endl;
//
//            Mat_<double> originalTransform(4,4,0.0);
//            R.copyTo(originalTransform(cv::Rect(0,0,3,3)));
//            originalTransform.at<double>(Point(3,0)) = (double) T.val[0];
//            originalTransform.at<double>(Point(3,1)) = (double) T.val[1];
//            originalTransform.at<double>(Point(3,2)) = (double) T.val[2];
//            originalTransform.at<double>(Point(3,3)) = 1.0;
//
//            cout << "orginal = " << originalTransform << endl;
//
//            Mat_<double> additionalTransform(4,4,0.0);
//            Rsupp.copyTo(additionalTransform(cv::Rect(0,0,3,3)));
//            additionalTransform.at<double>(Point(3,0)) = (double) Tsupp.val[0];
//            additionalTransform.at<double>(Point(3,1)) = (double) Tsupp.val[1];
//            additionalTransform.at<double>(Point(3,2)) = (double) Tsupp.val[2];
//            additionalTransform.at<double>(Point(3,3)) = 1.0;
//
//            cout << "addi = " << additionalTransform << endl;
//
//            Mat_<double> composedTransform = originalTransform*additionalTransform;
//
//            cout << "composed = " << composedTransform << endl;
//
//            composedTransform(cv::Rect(0,0,3,3)).copyTo(R);
//            T.val[0] = composedTransform.at<double>(Point(3,0));
//            T.val[1] = composedTransform.at<double>(Point(3,1));
//            T.val[2] = composedTransform.at<double>(Point(3,2));
//        }
//
//        cout << "R = " << R << endl;
//        cout << "T = " << T << endl;
//
//        FileStorage fsRawDepth;
//        fsRawDepth.open(rawDepthFile, FileStorage::READ);
//        Mat rawDepth;
//        fsRawDepth["rawDepth"] >> rawDepth;
//
//        PerformanceEvaluator evaluator(rawDepth, finalDisp, highGradPoints, kinectCamMatrix, kinectDistortion, Q, R, T,
//                                       commonROI.x, commonROI.y);

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}

// display point cloud -> DONE WITH MESHLAB
// add second image in test -> DONE

// kinect vs stereo error estimation -> DONE PerformanceEvaluator
// TODO make stereo module then calculate error over multiple images
// add support point to finalDisparities
// take into account the cropping made for SSE census!! and deal with the resize? -> NO NEED, the origin of the new ROI stays the same
// TODO investigate why huge error on some reprojected points

// TODO check influence of auto contrast techniques
// TODO fix & check the influence of census transform with variable window
// TODO finetune the parameters minimizing kinect mean error

// TODO more pictures (!! take new calibration images first)
// TODO test influence of exposure time and gain (& lightning conditions?)
// TODO test influence of bad calibration
// TODO make different plots of error data (mean error as function of depth, sigma as function of depth, dist error points in XZ plane for some images)
// TODO dist error curve as a function of the distance to a frontal plane

// build test dataset with ball and laser meter
// TODO add pictures with more object around the ball to the dataset
// TODO code to evaluate performance of ball localisation

// TODO replace cramer plane param calculation to increase speed (do it earlier to speed up testing?)
// TODO make it work on the robot platform
