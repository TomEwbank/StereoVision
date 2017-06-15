//
// Created by dallas on 14.06.17.
//

#ifndef PROJECT_BALLGROUNDTRUTH_H
#define PROJECT_BALLGROUNDTRUTH_H

#include <string>
#include <opencv2/opencv.hpp>

class BallGroundTruth {

public:

    // Ball ROI
    int x;
    int y;
    int width;
    int height;

    // Depth at wich the ball is located
    double depth;

    // Ball parameters
    float cx;
    float cy;
    float radius;

    // List of pixels belonging to the ball (circle circumscripted in the ROI)
    std::list<cv::Point2i> ballPixels;

    friend std::istream& operator>>(std::istream& str, BallGroundTruth& data);

    cv::Point2d getCoordInROI(cv::Rect roi) {
        return cv::Point2d(x-roi.x, y-roi.y);
    }

private:
    void computeBallPixels();

};


#endif //PROJECT_BALLGROUNDTRUTH_H
