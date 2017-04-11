//
// Created by dallas on 09.04.17.
//

#ifndef PROJECT_GROUNDTHRUTH_H
#define PROJECT_GROUNDTHRUTH_H

#include <string>
#include <opencv2/opencv.hpp>

class GroundThruth {

public:

    std::string pointName;
    int x;
    int y;
    float disparity;
    double distance;

    friend std::istream& operator>>(std::istream& str, GroundThruth& data);

    cv::Point2d getCoordInROI(cv::Rect roi) {
        return cv::Point2d(x-roi.x, y-roi.y);
    }

};


#endif //PROJECT_GROUNDTHRUTH_H
