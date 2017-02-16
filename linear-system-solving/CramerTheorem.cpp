#include "CramerTheorem.h"

CramerTheorem::CramerTheorem(unsigned long size, std::vector<std::vector<double>> coefficients) {
    std::vector<std::vector<double>> A_elements;
    std::vector<std::vector<double>> B_elements;
    for(int i = 0; i < size; ++i) {
        std::vector<double> A_row;
        std::vector<double> B_row;
        for(int j = 0; j < size + 1; ++j) {
            if(j == size) {
                B_row.push_back(coefficients[i][j]);
                break;
            }
            A_row.push_back(coefficients[i][j]);
        }
        A_elements.push_back(A_row);
        B_elements.push_back(B_row);
    }
    A = new Matrix(A_elements);
    B = new Matrix(B_elements);
    result = CramerResult::NOT_CALCULATED_YET;
}

void CramerTheorem::calculate() {
    std::vector<double> AB_determinants;
    double A_determinant = A->determinant();
    for(int i = 0; i < A->sizeY(); ++i) {
        Matrix mixed = this->mixAB(i);
        AB_determinants.push_back(mixed.determinant());
    }
    // 2 options - no solutions or infinite solutions
    // in this case
    if(A_determinant == 0.0) {
        result = CramerResult::INFINITE_RESULTS;
        for(double determinant : AB_determinants) {
            if(determinant != 0.0) {
                result = CramerResult::NO_RESULT;
                break;
            }
        }
        return;
    }
    // finite solutions
    result = CramerResult::FINITE_RESULTS;
    for(int i = 0; i < A->sizeY(); ++i) {
        results.push_back(AB_determinants[i] / A_determinant);
    }
}

Matrix CramerTheorem::mixAB(int index) {
    const std::vector<std::vector<double>> matrix = A->getVector();
    std::vector<std::vector<double>> new_matrix;
    for(int i = 0; i < matrix.size(); ++i) {
        std::vector<double> row;
        for(int j = 0; j < A->sizeX(); ++j) {
            row.push_back(j == index ? B->getVector()[i][0] : matrix[i][j]);
        }
        new_matrix.push_back(row);
    }
    return Matrix(new_matrix);
}

CramerResult CramerTheorem::getResult() {
    return result;
}

std::vector<double> CramerTheorem::getFiniteResults() {
    if(result == CramerResult::NOT_CALCULATED_YET) this->calculate();
    return results;
}