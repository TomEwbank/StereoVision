

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
        params.showImages = true;
        params.colorMapSliding = 60;

        // Generate groundTruth data
        String folderName = "imgs_rectified/";
        String serie = "ball_grassfloor";//"ball_woodfloor";//// "ball_grassfloor_light";//"ball_obstacles";//
        String groundTruthFile = folderName+"ROI_"+serie+".txt";
        ifstream readFile(groundTruthFile);
        vector<BallGroundTruth> groundTruthVec;
        BallGroundTruth data;
        while(data << readFile) {
            groundTruthVec.push_back(data);
        }

        String calibFile = folderName+"stereoCalib_2305_rotx008_nothingInv.yml";//"stereoParams_2906.yml";//"stereoParams_2205_rotx008.yml";//

//        String gain_exposure[] = {"50_5_","50_10_","20_50_","10_100_","1_200_","1_300_","1_400_"};
//        String gain_exposure[] = {"1_400_"};
//        String gain_exposure[] = {"1_300_"};
        String gain_exposure[] = {"1_200_"};
//        String gain_exposure[] = {"10_100_"};
//        String gain_exposure[] = {"20_50_"};
//        String gain_exposure[] = {"50_10_"};
//        String gain_exposure[] = {"50_5_"};

        for(String s : gain_exposure) {

            String pairName = s+serie;
            ofstream outputFile("ballErrors_" + pairName + ".txt");

            FileStorage fs;
            fs.open(calibFile, FileStorage::READ);
            Rect commonROI;
            fs["common_ROI"] >> commonROI;
            Mat Q;
            fs["Q"] >> Q;

            int iteration = 8;
            double e = 0;
            double t = 0;
            for (BallGroundTruth &groundTruth: groundTruthVec) {

                String imNumber = std::to_string(iteration);
                cout << "nb: " << iteration << endl;
                String leftFile = folderName + "left_" + pairName + "_" + imNumber + "_rectified.png";
                String rightFile = folderName + "right_" + pairName + "_" + imNumber + "_rectified.png";


                // Read input images
                cv::Mat_<unsigned char> leftImg, rightImg;
                leftImg = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
                rightImg = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);



//            float gamma = 0.8;
//            cout << "gamma is " << gamma << endl;
//
//            Mat dst;
//            GammaCorrection(leftImg, dst, gamma);
//
            // Show
//            imshow("src",   leftImg);
//
//            waitKey(0);


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
                t += elapsed.total_microseconds() / 1.0e6;


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

//            cout << "ball pix size = " << groundTruth.getBallPixels().size() << endl;
//            namedWindow("ball pix");
//            imshow("ball pix", leftImg);
//            waitKey();

                cout << "true depth = " << trueDepth << ", error = " << error << endl;
                outputFile << trueDepth << ", " << error << endl;
                e += abs(error);
                ++iteration;
            }
            cout << "mean error = " << e / (iteration - 1) << endl;
            outputFile.close();

            double fps = groundTruthVec.size()/t;
            cout << fps << "fps" << endl;
        }

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}

