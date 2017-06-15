//
// Created by dallas on 09.04.17.
//

#include "GroundTruth.h"
#include <string>
#include <iostream>
#include <sstream>
#include <regex>

std::istream& GroundTruth::operator>>(std::istream& str, GroundTruth& data)
{
    std::string line;
    std::string sname;
    std::string sx;
    std::string sy;
    std::string sdisp;
    std::string sdist;
    if (std::getline(str,line))
    {
        std::stringstream iss(line);
        if ( std::getline(iss, sx, ',')        &&
             std::getline(iss, sy, ',')         &&
             std::getline(iss, sdisp, ',')      &&
             std::getline(iss, sname, ',') &&
             std::getline(iss, sdist))
        {
            /* OK: All read operations worked */
            data.pointName = std::regex_replace(sname, std::regex("^ +| +$|( ) +"), "$1");
            data.x = std::stoi(sx);
            data.y = std::stoi(sy);
            data.disparity = std::stof(sdisp);
            data.distance = std::stod(sdist)*10;
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