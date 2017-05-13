#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
    FileStorage fs;
    fs.open("rawDepth__1.yml", FileStorage::READ);
    Mat rawDepth;
    fs["rawDepth"] >> rawDepth;
    std::cout << rawDepth.rows << " " << rawDepth.cols << std::endl;
    namedWindow("Gradient left");
    imshow("Gradient left", rawDepth);
    waitKey(0);

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