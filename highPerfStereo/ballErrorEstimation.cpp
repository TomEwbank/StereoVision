

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
        params.uniqueness = 0.5;
        params.maxDisp = 100;
        params.leftRightStep = 1;
        params.costAggrWindowSize = 11;
        params.gradThreshold = 70;//25; // [0,255], disparity will be computed only for points with a higher absolute gradient
        params.tLow = 2;
        params.tHigh = 10;
        params.nIters = 1;
        params.resizeFactor = 1;
        params.applyBlur = false;
        params.applyHistEqualization = true;
        params.blurSize = 3;

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
        params.showImages = false;
        params.colorMapSliding = 60;

        // Generate groundTruth data
        String folderName = "imgs_rectified/";
        String serie = "ball_obstacles";//"ball_grassfloor_light";//
        String groundTruthFile = folderName+"ROI_"+serie+".txt";
        ifstream readFile(groundTruthFile);
        vector<BallGroundTruth> groundTruthVec;
        BallGroundTruth data;
        while(data << readFile) {
            groundTruthVec.push_back(data);
        }

        String pairName = "1_200_"+serie;
        String calibFile = folderName+"stereoParams_2806_rotx008.yml"; //"stereoCalib_2305_rotx008_nothingInv.yml";//

        ofstream outputFile("ballErrors_"+pairName+".txt");

        FileStorage fs;
        fs.open(calibFile, FileStorage::READ);
        Rect commonROI;
        fs["common_ROI"] >> commonROI;
        Mat Q;
        fs["Q"] >> Q;

        int iteration = 1;
        double e = 0;
        for(BallGroundTruth& groundTruth: groundTruthVec) {

            String imNumber = std::to_string(iteration);
            String leftFile = folderName + "left_" + pairName + "_" + imNumber + "_rectified.png";
            String rightFile = folderName + "right_" + pairName + "_" + imNumber + "_rectified.png";

            std::vector<double> timeProfile;
            ptime lastTime;
            time_duration elapsed;

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


//            Mat_<float> finalDisp2(commonROI.height, commonROI.width, (float) 0);
//            vector<Point> highGradPoints2;
//            params.showImages = false;
//            highPerfStereo(leftImg(commonROI), rightImg(commonROI), params, finalDisp2, highGradPoints2);

//            for(int x=0; x < finalDisp.cols; ++x) {
//                for(int y=0; y < finalDisp.rows; ++y) {
//                    float d1 = finalDisp.at<float>(Point(x,y));
//                    float d2 = finalDisp2.at<float>(Point(x,y));
//                    if (d1 != d2)
//                        cout << "(" << x << "," << y << ")   " << d1 << " - " << d2 << endl;
//                }
//            }


//            if(highGradPoints.size() == highGradPoints2.size()) {
//                cout << "OK size grd pts" << endl;
//                for(int i=0; i<highGradPoints.size(); ++i) {
//                    Point p1 = highGradPoints[i];
//                    Point p2 = highGradPoints2[i];
//                    if (p1 != p2) {
//                        cout << p1 << "////" << p2 << endl;
//                    }
//                }
//            } else {
//                cout << "grd pts not OK" << endl;
//                cout << "size 1 = " << highGradPoints.size() << ", size 2 = " << highGradPoints2.size() << endl;
//            }



//            cv::Mat dst = finalDisp / params.maxDisp;
//            namedWindow("disparity map");
//            imshow("disparity map", dst);
//            waitKey();

            // Compute error with ball groundTruth
            double trueDepth = groundTruth.getDepth();
            double error = groundTruth.getDepthError(finalDisp, highGradPoints, commonROI, Q);
//            double error2 = groundTruth.getDepthError(finalDisp2, highGradPoints2, commonROI, Q);


            for (cv::Point2i p : groundTruth.getBallPixels()) {
//                    cout << p << endl;
                leftImg.at<uchar>(p) = 0;
            }

            cout << "ball pix size = " << groundTruth.getBallPixels().size() << endl;
            namedWindow("ball pix");
            imshow("ball pix", leftImg);
            waitKey();

            cout << "true depth = " << trueDepth << ", error = " << error << endl;
            outputFile << trueDepth << ", " << error << endl;
            e += error;
            ++iteration;
        }
        cout << "mean error = " << e/(iteration-1) << endl;
        outputFile.close();

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}

