//
// Created by dallas on 14.02.17.
//

#include "Plane.h"
#include <linear-system-solving/CramerTheorem.h>
#include <vector>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace std;
using namespace boost;
using namespace boost::posix_time;

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

Plane::Plane(const Triangle2* triangle, Mat_<float> depthMatrix) {
    unsigned long size = 3;
    std::vector<std::vector<double>> coefficients;
    for(unsigned long i = 0; i < size; ++i) {
        std::vector<double> row;
        Point2* p = triangle->getCorner(i);
        row.push_back(p->x());
        row.push_back(p->y());
        row.push_back(1.0);
        row.push_back(depthMatrix.at<float>(Point(p->x(),p->y())));
        coefficients.push_back(row);
    }


    std::cout << std::endl << "Calculating plane parameters..." << std::endl;
    ptime lastTime = microsec_clock::local_time();
    CramerTheorem calculator(size, coefficients);
    calculator.calculate();
    time_duration elapsed = (microsec_clock::local_time() - lastTime);
    cout << "Time of calculation: " << elapsed.total_microseconds()/1.0e6 << "s" << endl;
    switch(calculator.getResult()) {
        case CramerResult::NO_RESULT:
            std::cout << "No results (system is false)" << std::endl;
            break;
        case CramerResult::INFINITE_RESULTS:
            std::cout << "Infinite results (system is true for every real x-es)" << std::endl;
            break;
        case CramerResult::FINITE_RESULTS:
            std::cout << "Finite results:" << std::endl;
            std::vector<double> results = calculator.getFiniteResults();
            for(int i = 0; i < size; ++i) {
                std::cout << "x" << (i + 1) << " = " << results[i] << std::endl;
            }
            a = results[0];
            b = results[1];
            c = results[2];
            break;
    }
}

float Plane::getDepth(Point p) {
    return a*p.x + b*p.y + c;
}

float Plane::getDepth(Point2 p) {
    return a*p.x() + b*p.y() + c;
}