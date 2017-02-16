#include <iostream>
#include <cmath>
#include "Matrix.h"

// Implementation only for linear system solving
// purposes. It leaks features like adding,
// multiplying, transposing etc.

Matrix::Matrix(std::vector<std::vector<double>> elements) : matrix(elements) {
    size_y = matrix.size();
    size_x = matrix[0].size();
}

Matrix Matrix::makeMinor(int x, int y) {
    std::vector<std::vector<double>> new_matrix;
    for(int i = 0; i < size_y; ++i) {
        if(i != y) {
            std::vector<double> temporary;
            for(int j = 0; j < size_x; ++j) {
                if(j != x) {
                    temporary.push_back(matrix[i][j]);
                }
            }
            new_matrix.push_back(temporary);
        }
    }
    return Matrix(new_matrix);
}

double Matrix::cofactor(int x, int y) {
    if(size_x != size_y) {
        throw "Only n x n matrix has cofactor";
    }
    Matrix minor = this->makeMinor(x, y);
    return pow(-1, x + y) * minor.determinant();
}

double Matrix::determinant() {
    if(size_x != size_y) {
        throw "Only n x n matrix has determinant";
    }
    // Determinants are calculated by recursion so
    // we define 2x2 determinant in a simple way.
    if(size_x == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    double determinant = 0;
    for(int i = 0; i < size_x; ++i) {
        determinant += matrix[0][i] * this->cofactor(i, 0);
    }

    return determinant;
}

unsigned long Matrix::sizeX() {
    return size_x;
}

unsigned long Matrix::sizeY() {
    return size_y;
}

const std::vector<std::vector<double>> Matrix::getVector() {
    return matrix;
}