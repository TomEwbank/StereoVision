/*
 *  utility classes for the high performance semi-dense stereo matching
 *
 *  Developed in the context of the master thesis:
 *      "Efficient and precise stereoscopic vision for humanoid robots"
 *  Author: Tom Ewbank
 *  Institution: ULg
 *  Year: 2017
 */

#ifndef PROJECT_HIGHPERFSTEREOCLASSES_H
#define PROJECT_HIGHPERFSTEREOCLASSES_H

#include <opencv2/opencv.hpp>
#include <fade2d/Fade_2D.h>

using namespace std;
using namespace cv;
using namespace GEOM_FADE2D;


class StereoParameters {
public:
    // Stereo matching parameters
    double uniqueness;
    int maxDisp;
    int minDisp = 0;
    int leftRightStep;
    int costAggrWindowSize;
    uchar gradThreshold; // [0,255], disparity will be computed only for points with a higher absolute gradient
    char tLow;
    char tHigh;
    int nIters;
    bool applyBlur;
    bool applyHistEqualization;
    int blurSize;
    int rejectionMargin;
    unsigned int occGridSize;

    // Laplace parameters (gradient calculation)
    int kernelSize;
    int scale;
    int delta;
    int ddepth;

    // Feature detection parameters
    double adaptivity;
    int minThreshold;
    bool traceLines;
    int nbLines;
    int lineSize;
    bool invertRows;
    int nbRows;

    // Misc. parameters
    bool recordFullDisp;
    bool showImages;
    int colorMapSliding;
};

/**
 * Encapsulation of the Triangle2 class in a new data structure so that
 * a hash function can be defined
 */
struct MeshTriangle
{
    Triangle2* t;

    bool operator==(const MeshTriangle &other) const
    { return (t->getCorner(0) == other.t->getCorner(0)
              && t->getCorner(1) == other.t->getCorner(1)
              && t->getCorner(2) == other.t->getCorner(2));
    }
};

/**
 * Definition of a hash function for triangles
 */
namespace std {

    template <>
    struct hash<MeshTriangle>
    {
        // Compute a hash value for a triangle, combining the hash values of
        // the coordinates of its barycenter, using XOR and bit shifting
        std::size_t operator()(const MeshTriangle& k) const
        {
            using std::size_t;
            using std::hash;
            using std::string;

            Point2 b = k.t->getBarycenter();
            float x = b.x();
            float y = b.y();

            return ((hash<float>()(x)
                     ^ (hash<float>()(y) << 1)) >> 1);
        }
    };

}

/**
 * Class representing a 3D plane of equation a*u + b*v + c = d in the space (u,v,d).
 * u and v should be pixels coordinates and d is a disparity value.
 */
class Plane {
    float a, b, c; // plane parameters
public:

    /**
     * Default constructor
     * The plane parameters are set to 0
     */
    Plane();

    /**
     * Constructor specifying the plane parameters
     *
     * @param a
     * @param b
     * @param c
     */
    Plane(float a, float b, float c);

    /**
     * Constructor of the Plane defined by the 3 vertices of a triangle,
     * the coordinates u and v of the vertices are given by a 2D triangle,
     * and the 3rd coordinate d of each vertex is located at the point
     * (u,v) of a given matrix (disparity map).
     *
     * @param t - the triangle
     * @param disparities - the matrix
     */
    Plane(const Triangle2* t, Mat_<float> disparities);

    /**
     * Get the d coordinate of the point in the plane at coordinates (u,v)
     *
     * @param p - the point (u,v)
     *
     * @return the d coordinate
     */
    float getDepth(Point p);

    /**
     * Get the d coordinate of the point in the plane at coordinates (u,v)
     *
     * @param p - the point (u,v)
     *
     * @return the d coordinate
     */
    float getDepth(Point2 p);

};

/**
 * Class storing the information of a confident support point for the triangular mesh
 */
class ConfidentSupport
{
public:
    int x;
    int y;
    float disparity;
    char cost;

    ConfidentSupport();
    ConfidentSupport(int x, int y, float d, char cost);
};

/**
 * Class storing the information of a canditate point for dense epipolar matching
 */
class InvalidMatch {
public:
    int x;
    int y;
    char cost;

    InvalidMatch();
    InvalidMatch(int x, int y, char cost);
};

/**
 * Class regrouping the confident support points and the candidate points for epipolar matching.
 * Candidates for epipolar matching are also called invalid matches.
 */
class PotentialSupports
{
    unsigned int rows, cols;
    vector<ConfidentSupport> confidentSupports;
    vector<InvalidMatch> invalidMatches;

public:
    PotentialSupports(int height, int width, char tLow, char tHigh);

    unsigned int getOccGridHeight();
    unsigned int getOccGridWidth();

    /**
     * Set the pixel (x,y) as the confident support of the cell (u,v) in the occupancy grid
     *
     * @param u
     * @param v
     * @param x
     * @param y
     * @param dispartity
     * @param cost
     */
    void setConfidentSupport(int u, int v, int x, int y, float dispartity, char cost);

    /**
     * Set the pixel (x,y) as the invalid match of the cell (u,v) in the occupancy grid
     *
     * @param u
     * @param v
     * @param x
     * @param y
     * @param cost
     */
    void setInvalidMatch(int u, int v, int x, int y, char cost);

    /**
     * @param u
     * @param v
     *
     * @return the confident support point at position (u,v) in the occupancy grid
     */
    ConfidentSupport getConfidentSupport(int u, int v);

    /**
     * @param u
     * @param v
     *
     * @return the invalid match at position (u,v) in the occupancy grid
     */
    InvalidMatch getInvalidMatch(int u, int v);
};



#endif //PROJECT_HIGHPERFSTEREOCLASSES_H
