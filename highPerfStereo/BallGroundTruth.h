//
// Created by dallas on 14.06.17.
//

#ifndef PROJECT_BALLGROUNDTRUTH_H
#define PROJECT_BALLGROUNDTRUTH_H

#include <string>
#include <opencv2/opencv.hpp>

class BallGroundTruth {

private:

    // Ball ROI
    int x;
    int y;
    int width;
    int height;

    // Depth at which the ball is located
    double depth;

    // Ball parameters
    float cx;
    float cy;
    float radius;

    // List of pixels belonging to the ball (circle circumscripted in the ROI)
    std::list<cv::Point2i> ballPixels;
public:
    const std::list<cv::Point2i> &getBallPixels() const;

private:

    void computeBallPixels();

public:

    std::istream& operator<<(std::istream& str);

    double getDepth();

    double getDepthError(cv::Mat_<float> disparityMap,
                         std::vector<cv::Point> validDisparities,
                         cv::Rect roi,
                         cv::Mat_<float> perspTransform);


};


#endif //PROJECT_BALLGROUNDTRUTH_H
