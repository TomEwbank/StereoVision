

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
        params.maxDisp = 210;
        params.leftRightStep = 1;
        params.costAggrWindowSize = 11;
        params.gradThreshold = 80; // [0,255], disparity will be computed only for points with a higher absolute gradient
        params.tLow = 3;
        params.tHigh = 15;
        params.nIters = 1;
        params.resizeFactor = 1;
        params.applyBlur = false;
        params.applyHistEqualization = true;
        params.blurSize = 3;
        params.rejectionMargin = 10;
        params.occGridSize = 32;

        // Feature detection parameters
        params.adaptivity = 0.25;
        params.minThreshold = 4;
        params.traceLines = false;
        params.nbLines = 20;
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
        String serie = "plane";

        String pairName = ""+serie;
        String calibFile = folderName+"stereoParams_2806_windows_camUntouched.yml";

        ofstream outputFile("errorCurve.txt");

        FileStorage fs;
        fs.open(calibFile, FileStorage::READ);
        Rect commonROI;
        fs["common_ROI"] >> commonROI;
        Mat Q;
        fs["Q"] >> Q;

        int iteration = 1;
        double e = 0;
        std::vector<double> groundTruthVec = {4056,
                                              3854,
                                              3657,
                                              3450,
                                              3247,
                                              3047,
                                              2845,
                                              2646,
                                              2449,
                                              2253,
                                              2050,
                                              1859,
                                              1665,
                                              1454,
                                              1258,
                                              1053,
                                              861,
                                              658,
                                              465,
                                              252};
        std::vector<double> x1 = {568,
                          565,
                          559,
                          559,
                          555,
                          548,
                          547,
                          543,
                          538,
                          533,
                          514,
                          501,
                          489,
                          476,
                          458,
                          417,
                          405,
                          379,
                          478,
                          446};

        std::vector<double> y1 = {270,
                          264,
                          259,
                          257,
                          251,
                          244,
                          241,
                          233,
                          224,
                          216,
                          203,
                          189,
                          177,
                          148,
                          119,
                          126,
                          119,
                          119,
                          306,
                          275};
        std::vector<double> x2 = {644,
                          651,
                          651,
                          654,
                          657,
                          662,
                          662,
                          668,
                          674,
                          679,
                          674,
                          677,
                          684,
                          707,
                          718,
                          759,
                          769,
                          745,
                          650,
                          653};
        std::vector<double> y2 = {346,
                          350,
                          351,
                          352,
                          353,
                          358,
                          356,
                          358,
                          360,
                          362,
                          363,
                          365,
                          372,
                          379,
                          379,
                          468,
                          483,
                          485,
                          478,
                          482};

        int i = 0;
        for(double groundTruth: groundTruthVec) {

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

            int upX = x1[i]-commonROI.x;
            int upY = y1[i]-commonROI.y;
            int botX = x2[i]-commonROI.x;
            int botY = y2[i]-commonROI.y;
            ++i;

            // Generate pointCloud
            std::vector<Vec3d> vin2;
            for (Point coordInROI : highGradPoints) {

                float disp = finalDisp.at<float>(coordInROI);

                if (disp != 0 && coordInROI.x > upX && coordInROI.x < botX && coordInROI.y > upY && coordInROI.y < botY) {
                    Vec3d p(coordInROI.x + commonROI.x, coordInROI.y + commonROI.y, disp);
                    cout << p << endl;
                    vin2.push_back(p);
                }
            }
            cout << "size = " << vin2.size() << endl;

            std::vector<Vec3d> vout2(vin2.size());
            perspectiveTransform(vin2, vout2, Q);
            double z = 0;
            for(Vec3d p : vout2)
                z += p.val[2];
            z = z/vout2.size();

            double error = z-(groundTruth-38);

            cout << "true depth = " << groundTruth << ", z = " << z << ", error = " << error << endl;
            outputFile << groundTruth << ", " << error << endl;
            e += (abs(error)/groundTruth)*1000;
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

