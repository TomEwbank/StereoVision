//
// Created by dallas on 14.06.17.
//

#include <unordered_set>
#include "BallGroundTruth.h"
#include <unordered_set>

std::istream& BallGroundTruth::operator<<(std::istream& str)
{
    std::string line;
    std::string sx1;
    std::string sy1;
    std::string sx2;
    std::string sy2;
    std::string sdist;
    if (std::getline(str,line))
    {
        std::stringstream iss(line);
        if ( std::getline(iss, sx1, ',')        &&
             std::getline(iss, sy1, ',')         &&
             std::getline(iss, sx2, ',')      &&
             std::getline(iss, sy2, ',') &&
             std::getline(iss, sdist))
        {
            /* OK: All read operations worked */
            this->x = std::stoi(sx1);
            this->y = std::stoi(sy1);
            this->width = std::stoi(sx2)-this->x;
            this->height = std::stof(sy2)-this->y;

            if (this->width != this->height)
                std::cout << "Warning: Ball ROI not square!" << std::endl;

            this->depth = (std::stod(sdist)- 0.038)*1000;
            // 0.038 is the approximate length between the reference of
            // the laser measure and the plane of the camera sensor

            this->radius = this->width/2;
            this->cx = this->x+this->radius;
            this->cy = this->y+this->radius;

            this->computeBallPixels();
        }
        else
        {
            // One operation failed.
            // So set the state on the main stream
            // to indicate failure.
            str.setstate(std::ios::failbit);
        }
    }
    return str;
}

void BallGroundTruth::computeBallPixels() {
    double radiusSquared = radius*radius;
    for (int i = x; i < x+width; ++i)
    {
        for (int j = y; j < y+height; ++j)
        {
            int dx = i - cx;
            int dy = j - cy;
            double distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(i,j));
                continue;
            }

            ++dx;
            distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(i,j));
                continue;
            }

            ++dy;
            distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(i,j));
                continue;
            }

            --dx;
            distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(i,j));
                continue;
            }
        }
    }
}

double BallGroundTruth::getDepth() {
    return depth;
}

double BallGroundTruth::getDepthError(cv::Mat_<float> disparityMap,
                                      std::vector<cv::Point> validDisparities,
                                      cv::Rect roi,
                                      cv::Mat_<float> perspTransform) {

//    // TODO Convert vector to set to check in constant time if a disparity exists for a particular pixel
//    std::unordered_set<cv::Point> validDispSet(validDisparities.begin(), validDisparities.end());

    std::vector<cv::Vec3d> vIn;
    std::vector<cv::Vec3d> vOut;
    for (std::list<cv::Point2i>::iterator it = ballPixels.begin(); it != ballPixels.end(); ++it) {

        cv::Point realCoord;
        cv::Point coordInROI;
        bool pixelHasDisparity = false;
        for (const cv::Point& p: validDisparities) {
            coordInROI = p;
            realCoord.x = coordInROI.x+roi.x;
            realCoord.y = coordInROI.y+roi.y;
            if (realCoord == *it) {
                pixelHasDisparity =  true;
                break;
            }
        }

        if (pixelHasDisparity) {
            cv::Vec3d p(realCoord.x, realCoord.y, disparityMap.at<float>(coordInROI));
            vIn.push_back(p);
        }
    }
//    std::vector<cv::Vec3d> vOut(vIn.size());
    std::cout << ballPixels.size() << " ---- " << vIn.size() << " ---- " << vOut.size() << std::endl;
    cv::perspectiveTransform(vIn, vOut, perspTransform);

    double closestZ = -1;
    for (const cv::Vec3d& p: vOut) {
        double z = p.val[2];
        if (closestZ < 0 || z < closestZ)
            closestZ = z;
    }

    return closestZ-depth;

//    double zSum = 0;
//    for (const cv::Vec3d& p: vOut) {
//        zSum = zSum + p.val[2];
//    }
//
//    double meanZ = zSum/vOut.size();
//
//    return meanZ-depth;
}


//BallGroundTruth::BallGroundTruth(int x, int y, int width, int height, double depth, double radius, double cx,
//                                 double cy) {
//    this->x = x;
//    this->y = y;
//    this->width = width;
//    this->height = height;
//    this->depth = depth;
//    this->radius = radius;
//    this->cx = cx;
//    this->cy = cy;
//    computeBallPixels();
//}
//
//
//std::istream& operator>>(std::istream& str, BallGroundTruth& data)
//{
//    std::string line;
//    std::string sx1;
//    std::string sy1;
//    std::string sx2;
//    std::string sy2;
//    std::string sdist;
//    if (std::getline(str,line))
//    {
//        std::stringstream iss(line);
//        if ( std::getline(iss, sx1, ',')        &&
//             std::getline(iss, sy1, ',')         &&
//             std::getline(iss, sx2, ',')      &&
//             std::getline(iss, sy2, ',') &&
//             std::getline(iss, sdist))
//        {
//            /* OK: All read operations worked */
//            int x = std::stoi(sx1);
//            int y = std::stoi(sy1);
//            int width = std::stoi(sx2)-data.x;
//            int height = std::stof(sy2)-data.y;
//
//            if (width != height)
//                throw std::runtime_error("Error: Ball ROI not square!");
//
//            double depth = std::stod(sdist)*10;
//
//            double radius = data.width/2;
//            double cx = data.x+data.radius;
//            double cy = data.y+data.radius;
//
//            data = new BallGroundTruth(x,y,width,height,depth,radius,cx,cy);
//        }
//        else
//        {
//            // One operation failed.
//            // So set the state on the main stream
//            // to indicate failure.
//            str.setstate(std::ios::failbit);
//        }
//    }
//    return str;
//}