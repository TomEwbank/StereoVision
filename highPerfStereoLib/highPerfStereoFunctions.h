/*
 *  High performance semi-dense stereo matching
 *
 *  Based on the paper "High-Performance and Tunable Stereo Reconstruction"
 *  from Sudeep Pillai
 *
 *  Developed in the context of the master thesis:
 *      "Efficient and precise stereoscopic vision for humanoid robots"
 *  Author: Tom Ewbank
 *  Institution: ULg
 *  Year: 2017
 */

#ifndef PROJECT_HIGHPERFSTEREOFUNCTIONS_H
#define PROJECT_HIGHPERFSTEREOFUNCTIONS_H

#include "highPerfStereoClasses.h"
#include <unordered_map>

/**
 * Express a color from the HLS space to the BGR space
 *
 * @param h - hue
 * @param l - lightness
 * @param s - saturation
 * @return a vector containing the corresponding BGR values
 */
Vec3b HLS2BGR(float h, float l, float s);

/**
 * Trace a line with a color gradient in an image
 *
 * @param img - the image in which trace the line
 * @param start - the Point where the line starts
 * @param end - the Point where the line ends
 * @param c1 - the color at which the gradient starts
 * @param c2 - the color at which the gradient ends
 */
void line2(Mat& img, const Point& start, const Point& end,
           const Scalar& c1,   const Scalar& c2);

/**
 * Get the interpolated disparity of a point inside the triangular mesh
 *
 * @param dt - the triangular mesh
 * @param planeTable - a lookup storing the parameters of the planes constituting the triangular mesh
 * @param disparities - the matrix containing the disparities of the support points of the triangular mesh
 * @param x - the x coordinate of the point for which the disparity is wanted
 * @param y - the y coordinate of the point for which the disparity is wanted
 *
 * @return the interpolated disparity
 */
float getInterpolatedDisparity(Fade_2D &dt, unordered_map<MeshTriangle, Plane> &planeTable, Mat_<float> disparities, int x,
                               int y);

/**
 * Cost evaluation step of the stereo matching algorithm (see master thesis paper)
 *
 * @param censusLeft - census transform of the reference image
 * @param censusRight - census transform of the target image
 * @param highGradPts - set of high gradient points
 * @param disparities - interpolated disparities
 *
 * Output:
 * @param matchingCosts - matching costs associated to the interpolated disparities
 */
void costEvaluation(const Mat_<unsigned int>& censusLeft,
                    const Mat_<unsigned int>& censusRight,
                    const vector<Point>& highGradPts,
                    const Mat_<float>& disparities,
                    Mat_<char>& matchingCosts);
/**
 * Disparity refinement step of the stereo matching algorithm (see master thesis paper)
 *
 * @param highGradPts - set of high gradient points
 * @param disparities - interpolated disparities
 * @param matchingCosts - matching costs associated to the interpolated disparities
 * @param tLow - confident matching cost threshold
 * @param tHigh - bad matching cost threshold
 * @param occGridSize - size of a cell of the occupancy grid
 *
 * Output:
 * @param finalDisparities - final semi-dense disparity map
 * @param finalCosts - matching costs associated to the final semi-dense disparity map
 *
 * @return the set of the most confident support points and the candidates for dense epipolar matching
 */
PotentialSupports disparityRefinement(const vector<Point>& highGradPts,
                                      const Mat_<float>& disparities,
                                      const Mat_<char>& matchingCosts,
                                      const char tLow, const char tHigh,
                                      const unsigned int occGridSize,
                                      Mat_<float>& finalDisparities,
                                      Mat_<char>& finalCosts);

/**
 * Dense epipolar matching
 *
 * @param censusLeft - census transform of the reference image
 * @param censusRight - census transform of the target image
 * @param censusSize - size of the census window
 * @param costAggrWindowSize - size of the cost aggregation window
 * @param leftPoint - point of the reference image that needs to find a match
 * @param minDisparity - minimum disparity for the epipolar search
 * @param maxDisparity - maximum disparity for the epipolar search
 *
 * @return a confident support point resulting from the matching.
 * If the matching was not a success, the disparity of the returned confident
 * support will be equal to -1.
 */
ConfidentSupport epipolarMatching(const Mat_<unsigned int>& censusLeft,
                                  const Mat_<unsigned int>& censusRight,
                                  int censusSize, int costAggrWindowSize,
                                  InvalidMatch leftPoint,
                                  int minDisparity, int maxDisparity);
/**
 * Support resampling step of the stereo matching algorithm (see master thesis paper)
 *
 * @param mesh - the triangular mesh
 * @param ps - the set of the most confident support points and the candidates for dense epipolar matching
 * @param censusLeft - the census transform of the reference image
 * @param censusRight - the census transform of the target image
 * @param censusSize - the size of the census window
 * @param costAggrWindowSize - the size of the cost aggregation window
 * @param tLow - confident matching cost threshold
 * @param tHigh - bad matching cost threshold
 * @param minDisparity - minimum disparity for the epipolar search
 * @param maxDisparity - maximum disparity for the epipolar search
 *
 * Output:
 * @param disparities - interpolated disparities updated at the new support points
 */
void supportResampling(Fade_2D &mesh, PotentialSupports &ps,
                       const Mat_<unsigned int> &censusLeft,
                       const Mat_<unsigned int> &censusRight,
                       int censusSize, int costAggrWindowSize,
                       char tLow, char tHigh, int minDisp, int maxDisp,
                       Mat_<float> &disparities);
/**
 * High performance semi-dense stereo matching
 *
 * @param leftImg - reference image
 * @param rightImg - target image
 * @param parameters - parameters of the stereo matching
 *
 * Output:
 * @param disparityMap - the semi-dense disparity map
 * @param computedPoints - the set of high gradient points of the reference image
 */
void highPerfStereo(cv::Mat_<unsigned char> leftImg,
                    cv::Mat_<unsigned char> rightImg,
                    StereoParameters parameters,
                    Mat_<float> &disparityMap,
                    vector<Point> &computedPoints);


#endif //PROJECT_HIGHPERFSTEREOFUNCTIONS_H
