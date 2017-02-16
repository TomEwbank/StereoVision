#ifndef LINEAR_SYSTEM_SOLVER_CRAMERTHEOREM_H
#define LINEAR_SYSTEM_SOLVER_CRAMERTHEOREM_H

#include <vector>
#include "Matrix.h"

enum CramerResult {
    NOT_CALCULATED_YET,
    NO_RESULT,
    INFINITE_RESULTS,
    FINITE_RESULTS
};

class CramerTheorem {
private:
    CramerResult result;
    std::vector<double> results;
    Matrix *A, *B;
    Matrix mixAB(int index);
public:
    CramerTheorem(unsigned long size, std::vector<std::vector<double>> coefficients);
    void calculate();
    CramerResult getResult();
    std::vector<double> getFiniteResults();
};

#endif