

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


float GetGamma(Mat& src)
{
    CV_Assert(src.data);
    CV_Assert(src.depth() != sizeof(uchar));

    int height = src.rows;
    int width  = src.cols;
    long size  = height * width;

    //!< histogram
    float histogram[256] = {0};
    uchar pvalue = 0;
    MatIterator_<uchar> it, end;
    for( it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++ )
    {
        pvalue = (*it);
        histogram[pvalue]++;

    }

    int threshold = 0;       //otsu阈值
    long sum0 = 0, sum1 = 0; //前景的灰度总和和背景灰度总和
    long cnt0 = 0, cnt1 = 0; //前景的总个数和背景的总个数

    double w0 = 0, w1 = 0;   //前景和背景所占整幅图像的比例
    double u0 = 0, u1 = 0;   //前景和背景的平均灰度
    double u = 0;            //图像总平均灰度
    double variance = 0;     //前景和背景的类间方差
    double maxVariance = 0;  //前景和背景的最大类间方差

    int i, j;
    for(i = 1; i < 256; i++) //一次遍历每个像素
    {
        sum0 = 0;
        sum1 = 0;
        cnt0 = 0;
        cnt1 = 0;
        w0   = 0;
        w1   = 0;
        for(j = 0; j < i; j++)
        {
            cnt0 += histogram[j];
            sum0 += j * histogram[j];
        }

        u0 = (double)sum0 /  cnt0;
        w0 = (double)cnt0 / size;

        for(j = i ; j <= 255; j++)
        {
            cnt1 += histogram[j];
            sum1 += j * histogram[j];
        }

        u1 = (double)sum1 / cnt1;
        w1 = 1 - w0;                 // (double)cnt1 / size;

        u = u0 * w0 + u1 * w1;

        //variance =  w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
        variance =  w0 * w1 *  (u0 - u1) * (u0 - u1);

        if(variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }

    // convert threshold to gamma.
    float gamma = 0.0;
    gamma = threshold/255.0;

    // return
    return gamma;
}



void GammaCorrection(Mat& src, Mat& dst, float fGamma)
{
    CV_Assert(src.data);

    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));

    // build look up table
    unsigned char lut[256];
    for( int i = 0; i < 256; i++ )
    {
        lut[i] = saturate_cast<uchar>(pow((float)(i/255.0), fGamma) * 255.0f);
    }

    // case 1 and 3 for different channels
    dst = src.clone();
    const int channels = dst.channels();
    switch(channels)
    {
        case 1:
        {

            MatIterator_<uchar> it, end;
            for( it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++ )
                *it = lut[(*it)];

            break;
        }
        case 3:
        {

            MatIterator_<Vec3b> it, end;
            for( it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++ )
            {
                (*it)[0] = lut[((*it)[0])]; // B
                (*it)[1] = lut[((*it)[1])]; // G
                (*it)[2] = lut[((*it)[2])]; // R
            }
            break;

        }
    } // end for switch
}


int main(int argc, char** argv) {
    try {

        StereoParameters params;

        // Stereo matching parameters
        params.uniqueness = 0.5;
        params.maxDisp = 190;
        params.minDisp = 36;
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
        params.showImages = true;
        params.colorMapSliding = 60;

        // Generate groundTruth data
        String folderName = "imgs_rectified/";
        String serie = "ball_grassfloor_light";//"ball_woodfloor";//
        String groundTruthFile = folderName+"ROI_"+serie+".txt";
        ifstream readFile(groundTruthFile);
        vector<BallGroundTruth> groundTruthVec;
        BallGroundTruth data;
        while(data << readFile) {
            groundTruthVec.push_back(data);
        }

        String pairName = "1_200_"+serie;
        String calibFile = folderName+"stereoCalib_2305_rotx008_nothingInv.yml";//"stereoParams_2205_rotx008.yml";//"stereoParams_2906.yml"; //

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

//            float gamma = 0.8;
//            cout << "gamma is " << gamma << endl;
//
//            Mat dst;
//            GammaCorrection(leftImg, dst, gamma);
//
//            // Show
//            imshow("src",   leftImg);
//            imshow("dst",   dst);
//
//            waitKey(0);


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

//            cout << "ball pix size = " << groundTruth.getBallPixels().size() << endl;
//            namedWindow("ball pix");
//            imshow("ball pix", leftImg);
//            waitKey();

            cout << "true depth = " << trueDepth << ", error = " << error << endl;
            outputFile << trueDepth << ", " << error << endl;
            e += abs(error);
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

