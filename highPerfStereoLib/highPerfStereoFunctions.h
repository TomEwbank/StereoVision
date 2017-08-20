/*
 *  High performance stereo matching library
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


Vec3b HLS2BGR(float h, float l, float s);

void line2(Mat& img, const Point& start, const Point& end,
           const Scalar& c1,   const Scalar& c2);

float getInterpolatedDisparity(Fade_2D &dt, unordered_map<MeshTriangle, Plane> &planeTable, Mat_<float> disparities, int x,
                               int y);

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
                                  int censusSize, int costAggrWindowSize,
                                  InvalidMatch leftPoint,
                                  int minDisparity, int maxDisparity);

void supportResampling(Fade_2D &mesh,
                       PotentialSupports &ps,
                       const Mat_<unsigned int> &censusLeft,
                       const Mat_<unsigned int> &censusRight,
                       int censusSize, int costAggrWindowSize,
                       Mat_<float> &disparities,
                       char tLow, char tHigh,
                       int minDisp, int maxDisp);

void highPerfStereo(cv::Mat_<unsigned char> leftImg,
                    cv::Mat_<unsigned char> rightImg,
                    StereoParameters parameters,
                    Mat_<float> &disparityMap,
                    vector<Point> &computedPoints);


#endif //PROJECT_HIGHPERFSTEREOFUNCTIONS_H
