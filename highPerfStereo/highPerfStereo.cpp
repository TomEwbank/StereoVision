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

class PotentialSupports
{
    unsigned int rows, cols;
    vector<ConfidentSupport> confidentSupports;
    vector<InvalidMatch> invalidMatches;

public:
    PotentialSupports(int height, int width, char tLow, char tHigh) : rows(height), cols(width),
                                                                      confidentSupports(rows*cols, ConfidentSupport(0,0,0,tLow)),
                                                                      invalidMatches(rows*cols, InvalidMatch(0,0,tHigh))
    {}

    unsigned int getOccGridHeight() {
        return rows;
    }

    unsigned int getOccGridWidth() {
        return cols;
    }

    ConfidentSupport getConfidentSupport(int u, int v) {
        return confidentSupports[v * cols + u];
    }

    InvalidMatch getInvalidMatch(int u, int v) {
        return invalidMatches[v * cols + u];
    }

    void setConfidentSupport(int u, int v, int x, int y, float dispartity, char cost) {
        ConfidentSupport cs(x,y,dispartity,cost);
        confidentSupports[v * cols + u] = cs;
    }

    void setInvalidMatch(int u, int v, int x, int y, char cost) {
        InvalidMatch im(x,y,cost);
        invalidMatches[v * cols + u] = im;
    }
};


struct MeshTriangle
{
    Triangle2* t;

    bool operator==(const MeshTriangle &other) const
    { return (t->getCorner(0) == other.t->getCorner(0)
              && t->getCorner(1) == other.t->getCorner(1)
              && t->getCorner(2) == other.t->getCorner(2));
    }
};

namespace std {

    template <>
    struct hash<MeshTriangle>
    {
        std::size_t operator()(const MeshTriangle& k) const
        {
            using std::size_t;
            using std::hash;
            using std::string;

            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:

            Point2 b = k.t->getBarycenter();
            float x = b.x();
            float y = b.y();

            return ((hash<float>()(x)
                     ^ (hash<float>()(y) << 1)) >> 1);
        }
    };

}


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


void costEvaluation(const Mat_<unsigned int>& censusLeft,
                    const Mat_<unsigned int>& censusRight,
                    const vector<Point>& highGradPts,
                    const Mat_<float>& disparities,
                    Mat_<char>& matchingCosts) {

    HammingDistance h;
    vector<Point>::const_iterator i;
    for( i = highGradPts.begin(); i != highGradPts.end(); i++){
        float d = disparities.ptr<float>(i->y)[i->x];
        int xRight = i->x-(int)floor(d+0.5);

        if (xRight < 2)
            continue;
        else {
            unsigned int cl = censusLeft.ptr<unsigned int>(i->y)[i->x];
            unsigned int cr = censusRight.ptr<unsigned int>(i->y)[xRight];
            matchingCosts.ptr<char>(i->y)[i->x] = h.calculate(cl, cr);
        }
    }

}


PotentialSupports disparityRefinement(const vector<Point>& highGradPts,
                                       const Mat_<float>& disparities,
                                       const Mat_<char>& matchingCosts,
                                       const char tLow, const char tHigh,
                                       const unsigned int occGridSize,
                                       Mat_<float>& finalDisparities,
                                       Mat_<char>& finalCosts) {

    unsigned int occGridHeight = (unsigned int) (disparities.rows/occGridSize) + 1;
    unsigned int occGridWidth = (unsigned int) (disparities.cols/occGridSize) +1;
    // TODO: ensure no more bugs at limits of grid

    PotentialSupports ps(occGridHeight, occGridWidth, tLow, tHigh);

    vector<Point>::const_iterator it;
    for( it = highGradPts.begin(); it != highGradPts.end(); it++) {
        int u = it->x;
        int v = it->y;
        float d = disparities.ptr<float>(v)[u];
        char mc = matchingCosts.ptr<char>(v)[u];

        // Get occupancy grid indices for the considered point
        int i = u / occGridSize;
        int j = v / occGridSize;

        // If matching cost is lower than previous best final cost
        if (mc < finalCosts.ptr(v)[u]) {
            finalDisparities.ptr<float>(v)[u] = d;
            finalCosts.ptr<char>(v)[u] = mc;
        }

        // If matching cost is lower than previous best valid cost
        if (mc < tLow && mc < ps.getConfidentSupport(i,j).cost)
            ps.setConfidentSupport(i, j, u, v, d, mc);

        // If matching cost is higher than previous worst invalid cost
        if (mc > tHigh && mc > ps.getInvalidMatch(i,j).cost)
            ps.setInvalidMatch(i, j, u, v, mc);

    }

    return ps;
}

ConfidentSupport epipolarMatching(const Mat_<unsigned int>& censusLeft,
                                  const Mat_<unsigned int>& censusRight,
                                  int censusSize,
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
    int halfWindowSize = 16;
    int censusMargin = censusSize/2;
    for(int i = leftPoint.x; i>=halfWindowSize+censusMargin && i>(leftPoint.x-maxDisparity); --i) {
        int cost = 0;
        for (int m=-halfWindowSize; m<=halfWindowSize && leftPoint.y+m < censusLeft.rows; ++m) {

            if (leftPoint.y+m < 0)
                continue;

            const unsigned int* cl = censusLeft.ptr<unsigned int>(leftPoint.y+m);
            const unsigned int* cr = censusRight.ptr<unsigned int>(leftPoint.y+m);
            for (int n = -halfWindowSize; n <= halfWindowSize && i+n < censusLeft.cols; ++n) {
                cost += (int) h.calculate(cl[leftPoint.x+n], cr[i+n]);
            }
        }

        if(cost < minCost) {
            matchingX = i;
            minCost = cost;
        }
    }

    ConfidentSupport result(leftPoint.x, leftPoint.y, (float) (leftPoint.x-matchingX), h.calculate(censusLeft.ptr<unsigned int>(leftPoint.y)[leftPoint.x], censusRight.ptr<unsigned int>(leftPoint.y)[matchingX]));

    return result;
}

void supportResampling(Fade_2D &mesh,
                       PotentialSupports &ps,
                       const Mat_<unsigned int> &censusLeft,
                       const Mat_<unsigned int> &censusRight,
                       int censusSize,
                       Mat_<float> &disparities,
                       char tLow, char tHigh, int maxDisp) {

    unsigned int occGridHeight = ps.getOccGridHeight();
    unsigned int occGridWidth = ps.getOccGridWidth();
    for (unsigned int j = 0; j < occGridHeight; ++j) {
        for (unsigned int i = 0; i < occGridWidth; ++i) {

            // sparse epipolar stereo matching for invalid pixels and add them to support points
            InvalidMatch invalid = ps.getInvalidMatch(i,j);
            if (invalid.cost > tHigh) {
                ConfidentSupport newSupp = epipolarMatching(censusLeft, censusRight, censusSize, invalid, maxDisp);
                if (newSupp.cost<tLow) {
                    disparities.ptr<float>(newSupp.y)[newSupp.x] = newSupp.disparity;
                    Point2 p(newSupp.x, newSupp.y);
                    mesh.insert(p);
                }
            }

            // add confident pixels to support points
            ConfidentSupport newSupp = ps.getConfidentSupport(i,j);
            //cout << newSupp.x << " " << newSupp.y << endl;
            if (newSupp.cost < tLow) {
                disparities.ptr<float>(newSupp.y)[newSupp.x] = newSupp.disparity;
                Point2 p(newSupp.x, newSupp.y);
                mesh.insert(p);
            }
        }
    }
}

cv::Vec3b ConvertColor( cv::Vec3b src, int code)
{
    cv::Mat srcMat(1, 1, CV_8UC3 );
    *srcMat.ptr< cv::Vec3b >( 0 ) = src;

    cv::Mat resMat;
    cv::cvtColor( srcMat, resMat, code);

    return *resMat.ptr< cv::Vec3b >( 0 );
}

cv::Vec3b HLS2BGR(float h, float l, float s) {
    if (h < 0 || h >= 360 || l < 0 || l > 1 || s < 0 || s > 1)
        throw invalid_argument("invalid HLS parameters");

    float c = (1 - abs(2*l-1))*s;
    float x = c * (1 - abs(((int)h/60)%2 - 1));
    float m = l - c/2;

    float r,g,b;

    if (h < 60) {
        r = c;
        g = x;
        b = 0;
    } else if (h < 120) {
        r = x;
        g = c;
        b = 0;
    } else if (h < 180) {
        r = 0;
        g = c;
        b = x;
    } else if (h < 240) {
        r = 0;
        g = x;
        b = c;
    } else if (h < 300) {
        r = x;
        g = 0;
        b = c;
    } else {
        r = c;
        g = 0;
        b = x;
    }

    r = (r+m)*255;
    g = (g+m)*255;
    b = (b+m)*255;

    Vec3b color((uchar)b,(uchar)g,(uchar)r);
    return color;
}


int main(int argc, char** argv) {
    try {

        // Stereo matching parameters
        double uniqueness = 0.3;
        int maxDisp = 35;
        int leftRightStep = 2;
        uchar gradThreshold = 50; // [0,255], disparity will be computed only for points with a higher absolute gradient
        char tLow = 3;
        char tHigh = 8;
        int nIters = 8;
        double resizeFactor = 0.75;

        // Feature detection parameters
        double adaptivity = 0.5;
        int minThreshold = 3;
        int lineNb = 10;
        int lineSize = 4;

        // Parse arguments
        if(argc != 3 && argc != 4) {
            cout << "Usage: " << argv[0] << " LEFT-IMG RIGHT-IMG [CALIBRARION-FILE]" << endl;
            return 1;
        }
        char* leftFile = argv[1];
        char* rightFile = argv[2];
        char* calibFile = argc == 4 ? argv[3] : NULL;

//        Mat imgtest = imread(leftFile, CV_LOAD_IMAGE_ANYCOLOR);
//        // Show what you got
//        namedWindow("HLS");
//        imshow("HLS", imgtest);
//        waitKey(0);
//        cvtColor(imgtest, imgtest, CV_BGR2HLS);
//        // Show what you got
//        namedWindow("HLS");
//        imshow("HLS", imgtest);
//        waitKey(0);


        // Read input images
        cv::Mat_<unsigned char> leftImgInit, rightImgInit;
        leftImgInit = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImgInit = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);
        equalizeHist(leftImgInit, leftImgInit);
        equalizeHist(rightImgInit, rightImgInit);
        GaussianBlur(leftImgInit, leftImgInit, Size(5, 5), 0, 0 );
        GaussianBlur(rightImgInit, rightImgInit, Size(5, 5), 0, 0 );
        if(leftImgInit.data == NULL || rightImgInit.data == NULL)
            throw sparsestereo::Exception("Unable to open input images!");

        cv::Mat_<unsigned char> leftImg, rightImg;
        resize(leftImgInit, leftImg, Size(), resizeFactor, resizeFactor);
        resize(rightImgInit, rightImg, Size(), resizeFactor, resizeFactor);

        // Crop image so that SSE implementation won't crash
        //cv::Rect myROI(0,0,1232,1104);
        cv::Rect myROI(0,0,16*(leftImg.cols/16),16*(leftImg.rows/16));
        leftImg = leftImg(myROI);
        rightImg = rightImg(myROI);

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
        cout << "Time for gradient: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;

        // Show what you got
        namedWindow("Gradient left");
        imshow("Gradient left", abs_grd );
        waitKey(0);

        // Init disparity map
        Mat_<float> disparities(leftImg.rows, leftImg.cols, (float) 0);

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

        // Show what you got
        namedWindow("Gradient mask");
        imshow("Gradient mask", highGradMask );
        waitKey(0);

        // Horizontal lines tracing in images for better feature detection
        lastTime = microsec_clock::local_time();
        cv::Mat_<unsigned char> leftImgAltered, rightImgAltered;
        leftImgAltered = leftImg.clone();
        rightImgAltered = rightImg.clone();
        int stepSize = leftImgAltered.rows/lineNb;
        for(int k = stepSize/2; k<leftImgAltered.rows; k = k+stepSize) {
            for(int j = k; j<k+lineSize && j<leftImgAltered.rows; ++j) {
                leftImgAltered.row(j).setTo(0);
                rightImgAltered.row(j).setTo(0);
            }
        }
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for line drawing: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;

        // Show what you got
        namedWindow("left altered image");
        imshow("left altered image", leftImgAltered );
        waitKey(0);
        namedWindow("right altered image");
        imshow("right altered image", rightImgAltered );
        waitKey(0);

        // Load rectification data
        StereoRectification* rectification = NULL;
        if(calibFile != NULL)
            rectification = new StereoRectification(CalibrationResult(calibFile));

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

        // Highlight matches as colored boxes
        Mat_<Vec3b> screen(leftImg.rows, leftImg.cols);
        cvtColor(leftImg, screen, CV_GRAY2BGR);
//        cvtColor(screen, screen, CV_BGR2HLS);
//        namedWindow("BGR2HLS");
//        imshow("BGR2HLS", screen);
//        waitKey();

        for(int i=0; i<(int)correspondences.size(); i++) {
            double scaledDisp = (double)correspondences[i].disparity() / maxDisp;
            Vec3b color = HLS2BGR(scaledDisp*359, 0.5, 1);
            cout << "HLS returned = " << (int) color.val[0] << "," << (int) color.val[1] << "," << (int) color.val[2] << endl;
//            color = ConvertColor(color, CV_HLS2BGR);
//            cout << "RGB = " << (int) color.val[0] << "," << (int) color.val[1] << "," << (int) color.val[2] << endl;
//            if(scaledDisp > 0.5)
//                color = Vec3b(0, (1 - scaledDisp)*512, 255);
//            else color = Vec3b(0, 255, scaledDisp*512);

            rectangle(screen, correspondences[i].imgLeft->pt - Point2f(2,2),
                      correspondences[i].imgLeft->pt + Point2f(2, 2),
                      (Scalar) color, CV_FILLED);
        }

        // Display image and wait
        namedWindow("Sparse stereo");
        imshow("Sparse stereo", screen);
        waitKey();


        // Create the triangulation mesh & the color disparity map
        Fade_2D dt;

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

            //Iterate over the triangles to retreive all unique edges
            std::set<std::pair<Point2 *, Point2 *> > sEdges;
            std::vector<Triangle2 *> vAllDelaunayTriangles;
            dt.getTrianglePointers(vAllDelaunayTriangles);
            for (std::vector<Triangle2 *>::iterator it = vAllDelaunayTriangles.begin();
                 it != vAllDelaunayTriangles.end(); ++it) {
                Triangle2 *t(*it);
                for (int i = 0; i < 3; ++i) {
                    Point2 *p0(t->getCorner((i + 1) % 3));
                    Point2 *p1(t->getCorner((i + 2) % 3));
                    if (p0 > p1) std::swap(p0, p1);
                    sEdges.insert(std::make_pair(p0, p1));
                }
            }

            // Display mesh
            Mat_<Vec3b> mesh(leftImg.rows, leftImg.cols);
            cvtColor(leftImg, mesh, CV_GRAY2BGR);
            set<std::pair<Point2 *, Point2 *>>::const_iterator pos;

            for (pos = sEdges.begin(); pos != sEdges.end(); ++pos) {

                Point2 *p1 = pos->first;
                float scaledDisp = disparities.at<float>(Point(p1->x(), p1->y())) / maxDisp;
                Vec3b color1;
                if(scaledDisp > 0.5)
                    color1 = Vec3b(0, (1 - scaledDisp)*512, 255);
                else color1 = Vec3b(0, 255, scaledDisp*512);

                Point2 *p2 = pos->second;
                scaledDisp = disparities.at<float>(Point(p2->x(), p2->y())) / maxDisp;
                Vec3b color2;
                if(scaledDisp > 0.5)
                    color2 = Vec3b(0, (1 - scaledDisp)*512, 255);
                else color2 = Vec3b(0, 255, scaledDisp*512);


                line2(mesh, Point(p1->x(), p1->y()), Point(p2->x(), p2->y()), (Scalar) color1, (Scalar) color2);
            }

            // Display image and wait
            namedWindow("Triangular mesh");
            imshow("Triangular mesh", mesh);
            waitKey();

            // Init lookup table for plane parameters
            unordered_map<MeshTriangle, Plane> planeTable;

            // Disparity interpolation
            lastTime = microsec_clock::local_time();
            for (int j = 0; j < mesh.rows; ++j) {
                float *pixel = disparities.ptr<float>(j);
                for (int i = 0; i < mesh.cols; ++i) {
                    Point2 pointInPlaneFade = Point2(i, j);
                    //Point2f pointInPlaneCv = Point2f(i,j);
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
                        //disparities.at<float>(pointInPlaneCv) = plane.getDepth(pointInPlaneCv);
                        pixel[i] = plane.getDepth(pointInPlaneFade);
                    }

                }
            }
            elapsed = (microsec_clock::local_time() - lastTime);
            cout << "Time for building dipsarity map: " << elapsed.total_microseconds() / 1.0e6 << "s" << endl;
            cout << "plane table size: " << planeTable.size() << endl;

            // Display interpolated disparities
            cv::Mat dst = disparities/maxDisp;
            namedWindow("Full interpolated disparities");
            imshow("Full interpolated disparities", dst);
            waitKey();

            Mat outputImg;
            Mat temp = dst*255;
            temp.convertTo(outputImg, CV_8UC1);
            imwrite("disparity"+to_string(iter)+".png", outputImg);

            Mat_<char> matchingCosts(leftImg.rows, leftImg.cols, tHigh);
            costEvaluation(censusLeft, censusRight, highGradPoints, disparities, matchingCosts);
            PotentialSupports ps = disparityRefinement(highGradPoints, disparities, matchingCosts,
                                                        tLow, tHigh, occGridSize, finalDisp, finalCosts);

            // Highlight matches as colored boxes
            Mat_<Vec3b> badPts(leftImg.rows, leftImg.cols);
            cvtColor(leftImg, badPts, CV_GRAY2BGR);

            for(unsigned int i = 0; i<ps.getOccGridWidth(); ++i){
                for(unsigned int j=0; j<ps.getOccGridHeight(); ++j){
                    InvalidMatch p = ps.getInvalidMatch(i,j);
                    rectangle(badPts, Point2f(p.x,p.y) - Point2f(2,2),
                              Point2f(p.x,p.y) + Point2f(2, 2),
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

            if (iter != nIters) {

                // Support resampling
                supportResampling(dt, ps, censusLeft, censusRight, 5, disparities, tLow, tHigh, maxDisp);
                occGridSize = max((unsigned int) 1, occGridSize / 2);
            }
        }



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
