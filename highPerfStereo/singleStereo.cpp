

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
#include "highPerfStereoLib.h"
#include "BallGroundTruth.h"

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
        params.uniqueness = 0.4;
        params.maxDisp = 190;
        params.minDisp = 36;
        params.leftRightStep = 1;
        params.costAggrWindowSize = 11;
        params.gradThreshold = 100; // [0,255], disparity will be computed only for points with a higher absolute gradient
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
        params.adaptivity = 0.1;
        params.minThreshold = 4;
        params.traceLines = false;
        params.nbLines = 40;
        params.lineSize = 2;
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

        // Generate groundTruth data
        String folderName = "imgs_rectified/";
        String serie = "ball_grassfloor";//"ball_obstacles";//"ball_grassfloor_light";//

        String pairName = "1_300_"+serie;
        String calibFile = folderName+"stereoCalib_2305_rotx008_nothingInv.yml";//"stereoParams_2906.yml";//

        FileStorage fs;
        fs.open(calibFile, FileStorage::READ);
        Rect commonROI;
        fs["common_ROI"] >> commonROI;
        Mat Q;
        fs["Q"] >> Q;


        String imNumber = "11";
        String leftFile = folderName + "left_" + pairName + "_" + imNumber + "_rectified.png";
        String rightFile = folderName + "right_" + pairName + "_" + imNumber + "_rectified.png";

        // Read input images
        cv::Mat_<unsigned char> leftImg, rightImg;
        leftImg = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImg = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);


        Mat_<float> finalDisp(commonROI.height, commonROI.width, (float) 0);
        vector<Point> highGradPoints;

        if (leftImg.data == NULL || rightImg.data == NULL)
            throw sparsestereo::Exception("Unable to open input images!");

        ptime lastTime;
        time_duration elapsed;
        lastTime = microsec_clock::local_time();
        // Compute disparities
        highPerfStereo(leftImg(commonROI), rightImg(commonROI), params, finalDisp, highGradPoints);
        elapsed = (microsec_clock::local_time() - lastTime);
        double sec = elapsed.total_microseconds() / 1.0e6;
        double fps = 1/sec;
        cout << sec << "s ---> " << fps << " FPS" << endl;

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}

