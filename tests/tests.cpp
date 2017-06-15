#include "opencv2/opencv.hpp"

//#include <boost/accumulators/accumulators.hpp>
//#include <boost/accumulators/statistics.hpp>
//#include <boost/bind/placeholders.hpp>

using namespace std;
using namespace cv;
//using namespace boost;
//using namespace boost::accumulators;


int main(int, char**)
{
    FileStorage fsRawDepth;
    fsRawDepth.open("rawDepth_kinect_error_test_2.yml", FileStorage::READ);
    Mat rawDepth;
    fsRawDepth["rawDepth"] >> rawDepth;

    double depthValue = (double) rawDepth.at<int>(Point(301,274));
    double z = (1.0 / (depthValue * -0.0030711016 + 3.3309495161));
    cout << z << endl;


//    std::vector<float> distErrorMeans;
//    distErrorMeans.push_back(1.0);
//    distErrorMeans.push_back(2.0);
//    distErrorMeans.push_back(3.0);
//    accumulator_set<float, stats<tag::variance(lazy)>> mean_dist_acc;
////    mean_dist_acc = std::for_each(distErrorMeans.begin(), distErrorMeans.end(), mean_dist_acc);
//    std::cout << mean(mean_dist_acc) << std::endl;
//
//    std::vector<float> v;
//    v.push_back(3.0);
//    v.push_back(4.0);
//    v.push_back(5.0);
//    mean_dist_acc = std::for_each(v.begin(), v.end(), mean_dist_acc);
//    std::cout << mean(mean_dist_acc) << std::endl;


//    FileStorage fs;
//    fs.open("rawDepth__1.yml", FileStorage::READ);
//    Mat rawDepth;
//    fs["rawDepth"] >> rawDepth;
//    std::cout << rawDepth.rows << " " << rawDepth.cols << std::endl;
//    namedWindow("Gradient left");
//    imshow("Gradient left", rawDepth);
//    waitKey(0);

//    VideoCapture cap(0); // open the default camera
//    if(!cap.isOpened())  // check if we succeeded
//        return -1;
//
//    Mat edges;
//    namedWindow("edges",1);
//    for(;;)
//    {
//        Mat frame;
//        cap >> frame; // get a new frame from camera
//        //cvtColor(frame, edges, CV_BGR2GRAY);
//        imshow("edges", frame);
//        if(waitKey(30) >= 0) break;
//    }
//    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}