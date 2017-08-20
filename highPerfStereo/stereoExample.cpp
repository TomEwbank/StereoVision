/*
 *  Example of semi-dense disparity map computation
 *
 *  Developed in the context of the master thesis:
 *      "Efficient and precise stereoscopic vision for humanoid robots"
 *  Author: Tom Ewbank
 *  Institution: ULg
 *  Year: 2017
 */

#include <highPerfStereoLib/highPerfStereoFunctions.h>

using namespace std;

int main(int argc, char** argv) {
    try {

        StereoParameters params;

        // Stereo matching parameters
        params.uniqueness = 0.4;
        params.maxDisp = 100;
        params.minDisp = 36;
        params.leftRightStep = 1;
        params.costAggrWindowSize = 11;
        params.gradThreshold = 100; // [0,255], disparity will be computed only for points with a higher absolute gradient
        params.tLow = 2;
        params.tHigh = 6;
        params.nIters = 4;
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


        String folderName = "test_images/";
        String pairName = "test_image";
        String fileExtension = ".ppm";
        String calibFile = folderName+"stereoCalib.yml";


        String leftFile = folderName + "left_" + pairName + fileExtension;
        String rightFile = folderName + "right_" + pairName + fileExtension;

        // Read input images
        cv::Mat leftImg, rightImg, leftRectImg, rightRectImg, leftColorImg, leftColRectImg;
        // Load grayscale images to input in the stereo matching algorithm
        leftImg = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImg = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);
        // Load the left color images to color the 3D point cloud
        leftColorImg = imread(leftFile, CV_LOAD_IMAGE_COLOR);


        if (leftImg.data == NULL || rightImg.data == NULL)
            throw invalid_argument("Unable to open input images!");

        int imHeight = 460;
        int imWidth = 800;
        cv::Size size(imWidth, imHeight);
        cv::Size newSize(1037,631);

        // Read calibration parameters
        cv::Mat K1, K2;
        cv::Mat D1, D2;
        cv::Mat  R;
        cv::Vec3d T;

        FileStorage fs;
        fs.open(calibFile, FileStorage::READ);

        fs["K1"] >> K1; // Intrinsic parameters of left camera
        fs["K2"] >> K2; // Intrinsic parameters of right camera
        fs["D1"] >> D1; // Distortion parameters of right camera
        fs["D2"] >> D2; // Distortion parameters of right camera
        fs["R"] >> R; // Rotation between the cameras
        fs["T"] >> T; // Translation between the cameras

        fs.release();

        // Compute the rectification transforms and the disparty-depth mapping
        cv::Mat R1, R2, P1, P2, Q;
        cv::Rect validPixROI1(0,0,0,0), validPixROI2(0,0,0,0);
        cv::stereoRectify(K1, D1, K2, D2, size, R, T, R1, R2, P1, P2, Q, 0, -1, newSize, &validPixROI1, &validPixROI2);
        cv::Rect commonROI = validPixROI1 & validPixROI2;

        // Compute the rectification mapping
        cv::Mat lmapx, lmapy, rmapx, rmapy;
        cv::initUndistortRectifyMap(K1, D1, R1, P1, newSize, CV_32F, lmapx, lmapy);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, newSize, CV_32F, rmapx, rmapy);

        // Rectify the images
        cv::remap(leftImg, leftRectImg, lmapx, lmapy, cv::INTER_LINEAR);
        cv::remap(rightImg, rightRectImg, rmapx, rmapy, cv::INTER_LINEAR);
        cv::remap(leftColorImg, leftColRectImg, lmapx, lmapy, cv::INTER_LINEAR);

        // Init disparity map and vector of high gradient points
        cv::Mat_<float> finalDisp(commonROI.height, commonROI.width, (float) 0);
        vector<Point> highGradPoints;

        // Compute the disparity map
        highPerfStereo(leftRectImg(commonROI), rightRectImg(commonROI), params, finalDisp, highGradPoints);


        // Generate a 3D pointCloud file that can be read in a software like MeshLab
        std::vector<Vec3d> vin(highGradPoints.size());
        std::vector<Point>::iterator ptsIt;
        std::vector<Vec3d>::iterator vinIter;
        for (ptsIt = highGradPoints.begin(), vinIter = vin.begin();
             ptsIt < highGradPoints.end();
             ptsIt++, vinIter++) {

            Point coordInROI = *ptsIt;
            Vec3d p(coordInROI.x+commonROI.x, coordInROI.y+commonROI.y, finalDisp.at<float>(coordInROI));
            *vinIter = p;
        }

        std::vector<Vec3d> points3D(highGradPoints.size());
        perspectiveTransform(vin, points3D, Q);

        ofstream outputFile("pointCloud_"+pairName+".txt");
        std::vector<Vec3d>::iterator points3Diter;
        for (ptsIt = highGradPoints.begin(), points3Diter = points3D.begin();
             points3Diter < points3D.end();
             points3Diter++, ptsIt++) {

            Vec3d point3D = *points3Diter;
            Point pointInImage = *ptsIt;
            pointInImage.x += commonROI.x;
            pointInImage.y += commonROI.y;

            double x = point3D.val[0];
            double y = point3D.val[1];
            double z = point3D.val[2];

            Vec3b color = leftColRectImg.at<Vec3b>(pointInImage);
            double r = color.val[2];
            double g = color.val[1];
            double b = color.val[0];

            // Filter the generated 3D points, as not every high
            // gradient point might have received a disparity
            // + keep only the points whose depth is below 3 meters
            if (z > 0 && sqrt(pow(x,2)+pow(y,2)+pow(z,2)) < 3000)
                outputFile << x << " " << y << " " << z << " " << r << " " << g  << " " << b << endl;
        }
        outputFile.close();

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}
