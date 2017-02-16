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

using namespace std;
using namespace cv;
using namespace sparsestereo;
using namespace boost;
using namespace boost::posix_time;
using namespace GEOM_FADE2D;



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




int main(int argc, char** argv) {
	try {

		// Stereo matching parameters
		double uniqueness = 0.7;
		int maxDisp = 150;
		int leftRightStep = 2;
		
		// Feature detection parameters
		double adaptivity = 1.0;
		int minThreshold = 10;
		
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

        //cv::Rect myROI(0,0,1232,1104);
        cv::Rect myROI(0,0,16*(leftImg.cols/16),16*(leftImg.rows/16));
        leftImg = leftImg(myROI);
        rightImg = rightImg(myROI);

		
		if(leftImg.data == NULL || rightImg.data == NULL)
			throw sparsestereo::Exception("Unable to open input images!");

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

		ptime lastTime = microsec_clock::local_time();
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
				ImageConversion::unsignedToSigned(leftImg, &charLeft);
				Census::transform5x5(charLeft, &censusLeft);
				keypointsLeft.clear();
				leftFeatureDetector->detect(leftImg, keypointsLeft);
			}
			#pragma omp section
			{
				ImageConversion::unsignedToSigned(rightImg, &charRight);
				Census::transform5x5(charRight, &censusRight);
				keypointsRight.clear();
				rightFeatureDetector->detect(rightImg, keypointsRight);
			}
		}

		// Stereo matching. Not parallelized (overhead too large)
		stereo.match(censusLeft, censusRight, keypointsLeft, keypointsRight, &correspondences);

		
		// Print statistics
		time_duration elapsed = (microsec_clock::local_time() - lastTime);
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
        Mat_<float> disparities(leftImg.rows, leftImg.cols); // TODO have a data struc for disparity values
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

        // Display image and wait
        namedWindow("Stereo");
        imshow("Stereo", mesh);
        waitKey();


        // Init lookup table for plane parameters



		
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
