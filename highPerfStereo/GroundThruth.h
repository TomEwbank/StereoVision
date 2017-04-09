//
// Created by dallas on 09.04.17.
//

#ifndef PROJECT_GROUNDTHRUTH_H
#define PROJECT_GROUNDTHRUTH_H

#include <string>

class GroundThruth {

public:
    std::string pointName;
    int x;
    int y;
    float disparity;
    double distance;

    friend std::istream& operator>>(std::istream& str, GroundThruth& data);

};


#endif //PROJECT_GROUNDTHRUTH_H
