//
// Created by dallas on 14.02.17.
//

#ifndef PROJECT_PLANE_H
#define PROJECT_PLANE_H

#include <opencv2/opencv.hpp>
#include <fade2d/Triangle2.h>

using namespace GEOM_FADE2D;
using namespace cv;

class Plane {
        float a, b, c; // plane parameters for equation: a*u + b*v + c = d
    public:
        Plane();
        Plane(float, float, float);
        Plane(const Triangle2*, Mat_<float>);
        float getDepth(Point);
        float getDepth(Point2);

};


#endif //PROJECT_PLANE_H
