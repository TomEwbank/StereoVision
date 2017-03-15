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
#include <unordered_map>
#include <plane/Plane.h>
#include <math.h>
#include <algorithm>
#include "CensusTransform.h"

using namespace std;
using namespace cv;
using namespace sparsestereo;
using namespace boost;
using namespace boost::posix_time;
using namespace GEOM_FADE2D;

class ConfidentSupport
{
public:
    int x;
    int y;
    float disparity;
    char cost;

    ConfidentSupport() {
        x = 0;
        y = 0;
        disparity = 0;
        cost = 0;
    }

    ConfidentSupport(int x, int y, float d, char cost) {
        this->x = x;
        this->y = y;
        this->disparity = d;
        this->cost = cost;
    }
};

class InvalidMatch {
public:
    int x;
    int y;
    char cost;

    InvalidMatch() {
        x = 0;
        y = 0;
        cost = 0;
    }

    InvalidMatch(int x, int y, char cost) {
        this->x = x;
        this->y = y;
        this->cost = cost;
    }
};

void line2(Mat& img, const Point& start, const Point& end,
           const Scalar& c1,   const Scalar& c2) {
    LineIterator iter(img, start, end, 8);

    for (int i = 0; i < iter.count; i++, iter++) {
        double alpha = double(i) / iter.count;
        // note: using img.at<T>(iter.pos()) is faster, but
        // then you have to deal with mat type and channel number yourself
        img(Rect(iter.pos(), Size(1, 1))) = c1 * (1.0 - alpha) + c2 * alpha;
    }
}

ConfidentSupport epipolarMatching(const Mat_<unsigned int>& censusLeft,
                                  const Mat_<unsigned int>& censusRight,
                                  InvalidMatch leftPoint, int maxDisparity) {

//    const unsigned int *rightEpipolar = censusRight.ptr<unsigned int>(leftPoint.y);
//    HammingDistance h;
//    unsigned int censusRef = censusLeft.ptr<unsigned int>(leftPoint.y)[leftPoint.x];
//    int minCost = h.calculate(censusRef,rightEpipolar[leftPoint.x])+1;
//    int matchingX = leftPoint.x;
//    for(int i = leftPoint.x; i>=5 && i>(leftPoint.x-maxDisparity); --i) {
//        int cost = h.calculate(censusRef,rightEpipolar[i]);
//
//        if(cost < minCost) {
//            matchingX = i;
//            minCost = cost;
//        }
//    }
//
//    ConfidentSupport result = {leftPoint.x, leftPoint.y, (float)(leftPoint.x-matchingX), 0};
//    //cout  << ">> " << result.x << ", " << result.y << ", " << result.disparity << ", " << result.cost << endl;
//    return result;

    HammingDistance h;
    int minCost =  2147483647;//32*5*5;
    int matchingX = leftPoint.x;
    for(int i = leftPoint.x; i>=5 && i>(leftPoint.x-maxDisparity); --i) {
        int cost = 0;
        for (int m=-2; m<=2; ++m) {
            const unsigned int* cl = censusLeft.ptr<unsigned int>(leftPoint.y+m);
            const unsigned int* cr = censusRight.ptr<unsigned int>(leftPoint.y+m);
            for (int n = -2; n <= 2; ++n) {
                cost += (int) h.calculate(cl[leftPoint.x+n], cr[i+n]);
            }
        }

        if(cost < minCost) {
            matchingX = i;
            minCost = cost;
        }
    }

    ConfidentSupport result(leftPoint.x, leftPoint.y, (float) (leftPoint.x-matchingX), 0);

    return result;
}

ConfidentSupport epipolarMatching(CensusTransform& censusLeft,
                                  CensusTransform& censusRight,
                                  InvalidMatch leftPoint, int maxDisparity, int aggregationSize) {

//    const unsigned int *rightEpipolar = censusRight.ptr<unsigned int>(leftPoint.y);
//    HammingDistance h;
//    unsigned int censusRef = censusLeft.ptr<unsigned int>(leftPoint.y)[leftPoint.x];
//    int minCost = h.calculate(censusRef,rightEpipolar[leftPoint.x])+1;
//    int matchingX = leftPoint.x;
//    for(int i = leftPoint.x; i>=5 && i>(leftPoint.x-maxDisparity); --i) {
//        int cost = h.calculate(censusRef,rightEpipolar[i]);
//
//        if(cost < minCost) {
//            matchingX = i;
//            minCost = cost;
//        }
//    }
//
//    ConfidentSupport result = {leftPoint.x, leftPoint.y, (float)(leftPoint.x-matchingX), 0};
//    //cout  << ">> " << result.x << ", " << result.y << ", " << result.disparity << ", " << result.cost << endl;
//    return result;

    HammingDistance h;
    int minCost =  2147483647;//32*5*5;
    int matchingX = leftPoint.x;
    for(int i = leftPoint.x; i>=5 && i>(leftPoint.x-maxDisparity); --i) {
        int cost = 0;
        for (int m=-aggregationSize; m<=aggregationSize; ++m) {

            for (int n = -aggregationSize; n <= aggregationSize; ++n) {
                vector<unsigned int> cl = censusLeft.getCensus(leftPoint.x+n, leftPoint.y+m);
                vector<unsigned int> cr = censusRight.getCensus(i+n, leftPoint.y+m);

                for (int w = 0; w < censusLeft.getNWords(); ++w)
                    cost += (int) h.calculate(cl[w], cr[w]);
            }
        }

        if(cost < minCost) {
            matchingX = i;
            minCost = cost;
        }
    }

    ConfidentSupport result(leftPoint.x, leftPoint.y, (float) (leftPoint.x-matchingX), 0);

    return result;
}


int main(int argc, char** argv) {
    try {

        // Stereo matching parameters
        double uniqueness = 0.7;
        int maxDisp = 100;
        int leftRightStep = 2;
        double resizeFactor = 1;

        // Parse arguments
        if(argc != 3 && argc != 4) {
            cout << "Usage: " << argv[0] << " LEFT-IMG RIGHT-IMG [CALIBRARION-FILE]" << endl;
            return 1;
        }
        char* leftFile = argv[1];
        char* rightFile = argv[2];
        char* calibFile = argc == 4 ? argv[3] : NULL;

        // Read input images
        cv::Mat_<unsigned char> leftImgInit, rightImgInit;
        leftImgInit = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImgInit = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);
        if(leftImgInit.data == NULL || rightImgInit.data == NULL)
            throw sparsestereo::Exception("Unable to open input images!");

        cv::Mat_<unsigned char> leftImg, rightImg;
        resize(leftImgInit, leftImg, Size(), resizeFactor, resizeFactor);
        resize(rightImgInit, rightImg, Size(), resizeFactor, resizeFactor);

        // Crop image so that SSE implementation won't crash
        cv::Rect myROI(0,0,16*(leftImg.cols/16),16*(leftImg.rows/16));
        leftImg = leftImg(myROI);
        rightImg = rightImg(myROI);

        ptime lastTime = microsec_clock::local_time();


        // Load rectification data
        StereoRectification* rectification = NULL;
        if(calibFile != NULL)
            rectification = new StereoRectification(CalibrationResult(calibFile));

        // The stereo matcher. SSE Optimized implementation is only available for a 5x5 window
        SparseStereo<CensusWindow<5>, short> stereo(maxDisp, 1, uniqueness,
                                                    rectification, false, false, leftRightStep);


        // Objects for storing final and intermediate results
        cv::Mat_<char> charLeft(leftImg.rows, leftImg.cols),
                charRight(rightImg.rows, rightImg.cols);
        Mat_<unsigned int> censusLeft(leftImg.rows, leftImg.cols),
                censusRight(rightImg.rows, rightImg.cols);

        // This part can be parallelized with OMP
#pragma omp parallel sections default(shared) num_threads(2)
        {
#pragma omp section
            {
                ImageConversion::unsignedToSigned(leftImg, &charLeft);
                Census::transform5x5(charLeft, &censusLeft);
            }
#pragma omp section
            {
                ImageConversion::unsignedToSigned(rightImg, &charRight);
                Census::transform5x5(charRight, &censusRight);
            }
        }
        time_duration elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for census transform: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;

        // Init final disparity map
        Mat_<float> finalDisp(leftImg.rows, leftImg.cols, (float) 0);

        lastTime = microsec_clock::local_time();
        for(int j = 2; j<leftImg.rows-2; ++j) {
            float* fdisp = finalDisp.ptr<float>(j);

            for(int i=2; i<leftImg.cols-2; ++i) {
                InvalidMatch p = {i,j,0};
                ConfidentSupport cs = epipolarMatching(censusLeft, censusRight, p, maxDisp);
                //cout  << "<< " << cs.x << ", " << cs.y << ", " << cs.disparity << ", " << cs.cost << endl;
                fdisp[i] = cs.disparity;
            }
        }

        // Print statistics
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for stereo matching: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;


        // Display disparity map
        Mat dst = finalDisp / maxDisp;
        namedWindow("High gradient disparities");
        imshow("High gradient disparities", dst);
        waitKey();

        Mat outputImg;
        Mat temp = dst*255;
        temp.convertTo(outputImg, CV_8UC1);
        imwrite("fastCensus.png", outputImg);

//        Mat finalColorDisp(finalDisp.rows, finalDisp.cols, CV_8UC3, Scalar(0, 0, 0));
//        for (int y = 0; y < finalColorDisp.rows; ++y) {
//            Vec3b *colorPixel = finalColorDisp.ptr<Vec3b>(y);
//            float *pixel = dst.ptr<float>(y);
//            for (int x = 0; x < finalColorDisp.cols; ++x)
//                if (pixel[x] > 0) {
//                    Vec3b color;
//                    if (pixel[x] > 0.5)
//                        color = Vec3b((1 - pixel[x]) * 512, 0, 255);
//                    else color = Vec3b(0, 255, pixel[x] * 512);
//                    colorPixel[x] = color;
//                }
//        }
//
//        namedWindow("High gradient color disparities");
//        imshow("High gradient color disparities", finalColorDisp);
//        waitKey();

//        cv::Mat_<unsigned char> censusTest;
//        censusTest = imread("censusTest.png", CV_LOAD_IMAGE_GRAYSCALE);
//        cv::Mat_<char> charImg(censusTest.rows, censusTest.cols);
//        ImageConversion::unsignedToSigned(censusTest, &charImg);

        CensusTransform censusLeftBis(leftImg, 5);
        CensusTransform censusRightBis(rightImg, 5);
        cout << censusLeftBis.getWidth() << endl;
        cout << censusLeftBis.getHeight() << endl;
        cout << censusLeftBis.getMaskSize() << endl;
        cout << censusLeftBis.getNWords() << endl;
        cout << censusRightBis.getWidth() << endl;
        cout << censusRightBis.getHeight() << endl;
        cout << censusRightBis.getMaskSize() << endl;
        cout << censusRightBis.getNWords() << endl;
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for census transform BIS: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;

//        for (int j=0; j<charImg.rows; ++j) {
//            for (int i=0; i<charImg.cols; ++i) {
//                for (int n=0; n<census.getNWords(); ++n) {
//                    cout << "(" << i << ", " << j << ", " << n << ")" << " --- " << census.getCensus(i,j)[n] << endl;
//                }
//            }
//        }

        // Init final disparity map BIS
        Mat_<float> finalDispBis(leftImg.rows, leftImg.cols, (float) 0);

        lastTime = microsec_clock::local_time();
        for(int j = 2; j<leftImg.rows-2; ++j) {
            float* fdisp = finalDispBis.ptr<float>(j);

            for(int i=2; i<leftImg.cols-2; ++i) {
                InvalidMatch p = {i,j,0};
                ConfidentSupport cs = epipolarMatching(censusLeftBis, censusRightBis, p, maxDisp, 2);
                cout  << "<< " << cs.x << ", " << cs.y << ", " << cs.disparity << ", " << cs.cost << endl;
                fdisp[i] = cs.disparity;
            }
        }

        // Print statistics
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for stereo matching BIS: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;


        // Display disparity map
        Mat dstBis = finalDispBis / maxDisp;
        namedWindow("High gradient disparities");
        imshow("High gradient disparities", dstBis);
        waitKey();

        Mat outputImg2;
        Mat temp2 = dstBis*255;
        temp.convertTo(outputImg, CV_8UC1);
        imwrite("fastCensus.png", outputImg2);

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}
