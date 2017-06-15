//
// Created by dallas on 14.06.17.
//

#include "BallGroundTruth.h"

std::istream& BallGroundTruth::operator>>(std::istream& str, BallGroundTruth& data)
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
            data.x = std::stoi(sx1);
            data.y = std::stoi(sy1);
            data.width = std::stoi(sx2)-data.x;
            data.height = std::stof(sy2)-data.y;

            if (data.width != data.height)
                std::cout << "Warning: Ball ROI not square!" << std::endl;

            data.depth = std::stod(sdist)*10;

            data.radius = data.width/2;
            data.cx = data.x+data.radius;
            data.cy = data.y+data.radius;

            data.computeBallPixels();
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
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int dx = x - cx;
            int dy = y - cy;
            double distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(x,y));
                continue;
            }

            ++dx;
            distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(x,y));
                continue;
            }

            ++dy;
            distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(x,y));
                continue;
            }

            --dx;
            distanceSquared = dx * dx + dy * dy;

            if (distanceSquared <= radiusSquared)
            {
                ballPixels.push_back(cv::Point2i(x,y));
                continue;
            }
        }
    }
}