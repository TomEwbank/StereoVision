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

    // Depth at which the ball is located, in mm.
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

    /**
     * @return the real depth of the ball
     */
    double getDepth();

    /**
     * Computes the error (in mm) between the real depth of the ball and its approximation based
     * on a given disparity map and perspective transform.
     *
     * @param disparityMap - a semi dense disparity map
     * @param validDisparities - the coordinates of the points that have a value encoded in the disparity map,
     * expressed in the reference frame of the region of interest in which the disparity map has been calculated
     * @param roi - the region of interest mentioned above
     * @param perspTransform - the transformation matrix obtained from calibration, that allows to convert the
     * disparities into 3D world points
     *
     * @return the error e = depth_approx - real_depth
     */
    double getDepthError(cv::Mat_<float> disparityMap,
                         std::vector<cv::Point> validDisparities,
                         cv::Rect roi,
                         cv::Mat_<float> perspTransform);


};


#endif //PROJECT_BALLGROUNDTRUTH_H
