/*
 *  Developed in the context of the master thesis:
 *      "Efficient and precise stereoscopic vision for humanoid robots"
 *  Author: Tom Ewbank
 *  Institution: ULg
 *  Year: 2017
 */

#include "highPerfStereoClasses.h"
#include <Eigen/Dense>

using namespace Eigen;

/*** Plane class functions ***/

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

Plane::Plane(const Triangle2* triangle, const Mat_<float> disparities) {

    Matrix3f M;
    Vector3f n;
    for(unsigned long i = 0; i < 3; ++i) {
        std::vector<double> row;
        Point2* p = triangle->getCorner(i);
        M(i,0) = (float) p->x();
        M(i,1) = (float) p->y();
        M(i,2) = 1.0;
        n(i) = disparities.at<float>(Point(p->x(),p->y()));

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


/*** ConfidentSupport class functions ***/

ConfidentSupport::ConfidentSupport() {
    x = 0;
    y = 0;
    disparity = 0;
    cost = 0;
}

ConfidentSupport::ConfidentSupport(int x, int y, float d, char cost) {
    this->x = x;
    this->y = y;
    this->disparity = d;
    this->cost = cost;
}

/*** InvalidMatch class functions ***/

InvalidMatch::InvalidMatch() {
    x = 0;
    y = 0;
    cost = 0;
}

InvalidMatch::InvalidMatch(int x, int y, char cost) {
    this->x = x;
    this->y = y;
    this->cost = cost;
}


/*** PotentialSupports class functions ***/

PotentialSupports::PotentialSupports(int height, int width, char tLow, char tHigh) :
        rows(height), cols(width),
        confidentSupports(rows*cols, ConfidentSupport(0,0,0,tLow)),
        invalidMatches(rows*cols, InvalidMatch(0,0,tHigh)) {}

unsigned int PotentialSupports::getOccGridHeight() {
    return rows;
}

unsigned int PotentialSupports::getOccGridWidth() {
    return cols;
}

void PotentialSupports::setConfidentSupport(int u, int v, int x, int y, float dispartity, char cost) {
    ConfidentSupport cs(x,y,dispartity,cost);
    confidentSupports[v * cols + u] = cs;
}

void PotentialSupports::setInvalidMatch(int u, int v, int x, int y, char cost) {
    InvalidMatch im(x,y,cost);
    invalidMatches[v * cols + u] = im;
}

ConfidentSupport PotentialSupports::getConfidentSupport(int u, int v) {
    return confidentSupports[v * cols + u];
}

InvalidMatch PotentialSupports::getInvalidMatch(int u, int v) {
    return invalidMatches[v * cols + u];
}