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

using namespace std;
using namespace cv;
using namespace sparsestereo;
using namespace boost;
using namespace boost::posix_time;
using namespace GEOM_FADE2D;


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



void costEvaluation(const Mat_<unsigned int>& censusLeft, const Mat_<unsigned int>& censusRight,
                    const vector<Point>& highGradPts, const Mat_<float>& disparities, Mat_<char>& matchingCosts) {
    HammingDistance h;
    vector<Point>::const_iterator i;
    for( i = highGradPts.begin(); i != highGradPts.end(); i++){
        float d = disparities.ptr<float>(i->y)[i->x];
        unsigned int cl = censusLeft.ptr<float>(i->y)[i->x];
        unsigned int cr = censusRight.ptr<float>(i->y)[i->x-(int)floor(d+0.5)];
        matchingCosts.ptr<float>(i->y)[i->x] = h.calculate(cl,cr);
    }

}


int main(int argc, char** argv) {
    try {

        // Stereo matching parameters
        double uniqueness = 0.7;
        int maxDisp = 100;
        int leftRightStep = 2;
        uchar gradThreshold = 128; // [0,255], disparity will be computed only for points with a higher absolute gradient

        // Feature detection parameters
        double adaptivity = 1.0;
        int minThreshold = 10;
        int lineNb = 40;
        int lineSize = 2;

        // Parse arguments
        if(argc != 3 && argc != 4) {
            cout << "Usage: " << argv[0] << " LEFT-IMG RIGHT-IMG [CALIBRARION-FILE]" << endl;
            return 1;
        }
        char* leftFile = argv[1];
        char* rightFile = argv[2];
        char* calibFile = argc == 4 ? argv[3] : NULL;

        // Read input images
        cv::Mat_<unsigned char> leftImg, rightImg;
        leftImg = imread(leftFile, CV_LOAD_IMAGE_GRAYSCALE);
        rightImg = imread(rightFile, CV_LOAD_IMAGE_GRAYSCALE);
        if(leftImg.data == NULL || rightImg.data == NULL)
            throw sparsestereo::Exception("Unable to open input images!");

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
        Mat_<float> disparities(leftImg.rows, leftImg.cols);

        // Get the set of high gradient points
        vector<Point> highGradPoints;
        int v = 0;
        Mat highGradMask(grd.rows, grd.cols, CV_8U, Scalar(0));
        for(int j = 0; j < abs_grd.rows; ++j) {
            uchar* pixel = abs_grd.ptr(j);
            for (int i = 0; i < abs_grd.cols; ++i) {
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
            for(int j = k; j<k+lineSize; ++j) {
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

        for(int i=0; i<(int)correspondences.size(); i++) {
            double scaledDisp = (double)correspondences[i].disparity() / maxDisp;
            Vec3b color;
            if(scaledDisp > 0.5)
                color = Vec3b(0, (1 - scaledDisp)*512, 255);
            else color = Vec3b(0, 255, scaledDisp*512);

            rectangle(screen, correspondences[i].imgLeft->pt - Point2f(2,2),
                      correspondences[i].imgLeft->pt + Point2f(2, 2),
                      (Scalar) color, CV_FILLED);
        }

        // Display image and wait
        namedWindow("Stereo");
        imshow("Stereo", screen);
        waitKey();


        // Create the triangulation mesh & the color disparity map
        Fade_2D dt;
        Mat colorDisparities(leftImg.rows, leftImg.cols, CV_8UC3, Scalar(0,0,0));

        for(int i=0; i<(int)correspondences.size(); i++) {
            float x = correspondences[i].imgLeft->pt.x;
            float y = correspondences[i].imgLeft->pt.y;
            float d = correspondences[i].disparity();

            disparities.at<float>(Point(x,y)) = d;

            Point2 p(x, y);
            dt.insert(p);

            double scaledDisp = (double) d / maxDisp;
            Vec3b color;
            if(scaledDisp > 0.5)
                color = Vec3b(0, (1 - scaledDisp)*512, 255);
            else color = Vec3b(0, 255, scaledDisp*512);
            colorDisparities.at<Vec3b>(Point(x,y)) = color;
        }

        //Iterate over the triangles to retreive all unique edges
        std::set<std::pair<Point2*,Point2*> > sEdges;
        std::vector<Triangle2*> vAllDelaunayTriangles;
        dt.getTrianglePointers(vAllDelaunayTriangles);
        for(std::vector<Triangle2*>::iterator it=vAllDelaunayTriangles.begin();it!=vAllDelaunayTriangles.end();++it)
        {
            Triangle2* t(*it);
            for(int i=0;i<3;++i)
            {
                Point2* p0(t->getCorner((i+1)%3));
                Point2* p1(t->getCorner((i+2)%3));
                if(p0>p1) std::swap(p0,p1);
                sEdges.insert(std::make_pair(p0,p1));
            }
        }

        // Display mesh
        Mat_<Vec3b> mesh(leftImg.rows, leftImg.cols);
        cvtColor(leftImg, mesh, CV_GRAY2BGR);
        set<std::pair<Point2*,Point2*>>::const_iterator pos;

        for(pos = sEdges.begin(); pos != sEdges.end(); ++pos) {

            Point2* p1 = pos->first;
            Vec3b color1 = colorDisparities.at<Vec3b>(Point(p1->x(),p1->y()));

            Point2* p2 = pos->second;
            Vec3b color2 = colorDisparities.at<Vec3b>(Point(p2->x(),p2->y()));

            line2(mesh, Point(p1->x(),p1->y()), Point(p2->x(),p2->y()), (Scalar) color1, (Scalar) color2);
        }


        rectangle(mesh, Point2f(700,300)-Point2f(2,2), Point2f(700,300)+Point2f(2,2), (Scalar) Vec3b(255, 0, 0), CV_FILLED);

        // Plane unit testing
        Point2 pointInPlaneFade = Point2(700,300);
        Point2f pointInPlaneCv = Point2f(700,300);

        lastTime = microsec_clock::local_time();
        Triangle2* t = dt.locate(pointInPlaneFade);
        elapsed = (microsec_clock::local_time() - lastTime);
        cout << "Time for triangle locate: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;

        for (int i=0; i<3; ++i) {
            Point2f p = Point2f(t->getCorner(i)->x(),t->getCorner(i)->y());
            cout << "p(" << t->getCorner(i)->x() << ", " << t->getCorner(i)->y() << ", "<< disparities.at<float>(p) << ")" << endl;
            rectangle(mesh, p-Point2f(2,2), p+Point2f(2,2), (Scalar) Vec3b(255, 0, 0), CV_FILLED);
        }

        Plane plane = Plane(t, disparities);
        float disp_700_300 = plane.getDepth(pointInPlaneCv);
        cout << "disp(700,300) = " << disp_700_300 << endl;

        double scaledDisp = (double)disp_700_300 / maxDisp;
        Vec3b color;
        if(scaledDisp > 0.5)
            color = Vec3b(0, (1 - scaledDisp)*512, 255);
        else color = Vec3b(0, 255, scaledDisp*512);

        rectangle(mesh, pointInPlaneCv-Point2f(2,2), pointInPlaneCv+Point2f(2,2), (Scalar) color, CV_FILLED);

        // Display image and wait
        namedWindow("Stereo");
        imshow("Stereo", mesh);
        waitKey();


        // Init lookup table for plane parameters
        unordered_map<MeshTriangle, Plane> planeTable;

        lastTime = microsec_clock::local_time();
        for (int j = 0; j<mesh.rows; ++j) {
            float* pixel = disparities.ptr<float>(j);
            for (int i = 0; i<mesh.cols; ++i) {
                Point2 pointInPlaneFade = Point2(i,j);
                //Point2f pointInPlaneCv = Point2f(i,j);
                Triangle2* t = dt.locate(pointInPlaneFade);
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
        cout << "Time for building dipsarity map: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;
        cout << "plane table size: " << planeTable.size() << endl;

        // Display first interpolated disparities and wait
        namedWindow("Stereo");
        cv::Mat dst;
        cv::normalize(disparities, dst, 0, 1, cv::NORM_MINMAX);
        imshow("Stereo", dst);
        waitKey();


        // Clean up
        delete leftFeatureDetector;
        delete rightFeatureDetector;
        if(rectification != NULL)
            delete rectification;

        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}
