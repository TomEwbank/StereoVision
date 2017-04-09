//
// Created by dallas on 09.04.17.
//

#ifndef PROJECT_HIGHPERFSTEREOLIB_H
#define PROJECT_HIGHPERFSTEREOLIB_H


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
#include <plane/Plane.h>
#include <math.h>
#include <algorithm>
#include <exception>

using namespace std;
using namespace cv;
using namespace sparsestereo;
using namespace boost;
using namespace boost::posix_time;
using namespace GEOM_FADE2D;


class ConfidentSupport
{
public:
    int x;
    int y;
    float disparity;
    char cost;

    ConfidentSupport() {
        x = 0;
        y = 0;
        disparity = 0;
        cost = 0;
    }

    ConfidentSupport(int x, int y, float d, char cost) {
        this->x = x;
        this->y = y;
        this->disparity = d;
        this->cost = cost;
    }
};

class InvalidMatch {
public:
    int x;
    int y;
    char cost;

    InvalidMatch() {
        x = 0;
        y = 0;
        cost = 0;
    }

    InvalidMatch(int x, int y, char cost) {
        this->x = x;
        this->y = y;
        this->cost = cost;
    }
};

class PotentialSupports
{
    unsigned int rows, cols;
    vector<ConfidentSupport> confidentSupports;
    vector<InvalidMatch> invalidMatches;

public:
    PotentialSupports(int height, int width, char tLow, char tHigh) : rows(height), cols(width),
                                                                      confidentSupports(rows*cols, ConfidentSupport(0,0,0,tLow)),
                                                                      invalidMatches(rows*cols, InvalidMatch(0,0,tHigh))
    {}

    unsigned int getOccGridHeight() {
        return rows;
    }

    unsigned int getOccGridWidth() {
        return cols;
    }

    ConfidentSupport getConfidentSupport(int u, int v) {
        return confidentSupports[v * cols + u];
    }

    InvalidMatch getInvalidMatch(int u, int v) {
        return invalidMatches[v * cols + u];
    }

    void setConfidentSupport(int u, int v, int x, int y, float dispartity, char cost) {
        ConfidentSupport cs(x,y,dispartity,cost);
        confidentSupports[v * cols + u] = cs;
    }

    void setInvalidMatch(int u, int v, int x, int y, char cost) {
        InvalidMatch im(x,y,cost);
        invalidMatches[v * cols + u] = im;
    }
};


struct MeshTriangle
{
    Triangle2* t;

    bool operator==(const MeshTriangle &other) const
    { return (t->getCorner(0) == other.t->getCorner(0)
              && t->getCorner(1) == other.t->getCorner(1)
              && t->getCorner(2) == other.t->getCorner(2));
    }
};

namespace std {

    template <>
    struct hash<MeshTriangle>
    {
        std::size_t operator()(const MeshTriangle& k) const
        {
            using std::size_t;
            using std::hash;
            using std::string;

            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:

            Point2 b = k.t->getBarycenter();
            float x = b.x();
            float y = b.y();

            return ((hash<float>()(x)
                     ^ (hash<float>()(y) << 1)) >> 1);
        }
    };

}


void line2(Mat& img, const Point& start, const Point& end,
           const Scalar& c1,   const Scalar& c2);


void costEvaluation(const Mat_<unsigned int>& censusLeft,
                    const Mat_<unsigned int>& censusRight,
                    const vector<Point>& highGradPts,
                    const Mat_<float>& disparities,
                    Mat_<char>& matchingCosts);


PotentialSupports disparityRefinement(const vector<Point>& highGradPts,
                                      const Mat_<float>& disparities,
                                      const Mat_<char>& matchingCosts,
                                      const char tLow, const char tHigh,
                                      const unsigned int occGridSize,
                                      Mat_<float>& finalDisparities,
                                      Mat_<char>& finalCosts);

ConfidentSupport epipolarMatching(const Mat_<unsigned int>& censusLeft,
                                  const Mat_<unsigned int>& censusRight,
                                  int censusSize,
                                  InvalidMatch leftPoint, int maxDisparity);

void supportResampling(Fade_2D &mesh,
                       PotentialSupports &ps,
                       const Mat_<unsigned int> &censusLeft,
                       const Mat_<unsigned int> &censusRight,
                       int censusSize,
                       Mat_<float> &disparities,
                       char tLow, char tHigh, int maxDisp);

cv::Vec3b ConvertColor( cv::Vec3b src, int code);

cv::Vec3b HLS2BGR(float h, float l, float s);



#endif //PROJECT_HIGHPERFSTEREOLIB_H
