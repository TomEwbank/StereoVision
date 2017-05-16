#include "opencv2/opencv.hpp"

//#include <boost/accumulators/accumulators.hpp>
//#include <boost/accumulators/statistics.hpp>
//#include <boost/bind/placeholders.hpp>

using namespace std;
using namespace cv;
//using namespace boost;
//using namespace boost::accumulators;

cv::Vec3b HLS2BGR(float h, float l, float s) {
    if (h < 0 || h >= 360 || l < 0 || l > 1 || s < 0 || s > 1)
        throw invalid_argument("invalid HLS parameters");

    float c = (1 - abs(2*l-1))*s;
//    float hh = h/60;
    float x = c*(1-abs(std::fmod(h/60,2)-1));
    float m = l - c/2;

    cout << "-----" <<  endl << c << " " << x << " " << m << endl;

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


    cout  << r << " " << g << " " << b << endl;

    Vec3b color((uchar)b,(uchar)g,(uchar)r);

    return color;
}


int main(int, char**)
{

    Mat_<Vec3b> colorMap(10,360);
    for (int i=0; i<360; ++i) {
        Mat_<Vec3b> color(10,1, HLS2BGR(i,0.5,1));
        color.copyTo(colorMap(Rect(i,0,1,10)));
    }
    namedWindow("Gradient left");
    imshow("Gradient left", colorMap);
    waitKey(0);


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