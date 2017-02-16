#ifndef LINEAR_SYSTEM_SOLVER_MATRIX_H
#define LINEAR_SYSTEM_SOLVER_MATRIX_H

#include <vector>

class Matrix {
private:
    const std::vector<std::vector<double>> matrix;
    unsigned long size_x, size_y;
public:
    Matrix(std::vector<std::vector<double>>);

    Matrix makeMinor(int x, int y);
    double cofactor(int x, int y);
    double determinant();
    unsigned long sizeX();
    unsigned long sizeY();
    const std::vector<std::vector<double>> getVector();
};

#endif //LINEAR_SYSTEM_SOLVER_MATRIX_H