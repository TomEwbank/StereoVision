//
// Created by dallas on 14.02.17.
//

#include "Plane.h"
#include <linear-system-solving/CramerTheorem.h>
#include <vector>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace boost;
using namespace boost::posix_time;
using namespace Eigen;


Plane::Plane() {
    a = 0;
    b = 0;
    c = 0;
}

Plane::Plane(float a, float b, float c) {
    this->a = a;
    this->b = b;
    this->c = c;
}

Plane::Plane(const Triangle2* triangle, const Mat_<float> depthMatrix) {

    Matrix3f M;
    Vector3f n;
    for(unsigned long i = 0; i < 3; ++i) {
        std::vector<double> row;
        Point2* p = triangle->getCorner(i);
        M(i,0) = (float) p->x();
        M(i,1) = (float) p->y();
        M(i,2) = 1.0;
        n(i) = depthMatrix.at<float>(Point(p->x(),p->y()));

    }

    Vector3f x = M.colPivHouseholderQr().solve(n);
    a = x(0);
    b = x(1);
    c = x(2);
}

float Plane::getDepth(Point p) {
    return a*p.x + b*p.y + c;
}

float Plane::getDepth(Point2 p) {
    return a*p.x() + b*p.y() + c;
}