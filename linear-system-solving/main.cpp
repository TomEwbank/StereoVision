#include <iostream>
#include <vector>
#include "CramerTheorem.h"

int main() {
    std::cout << "+------------------------------------------------+" << std::endl;
    std::cout << "+ linear system solver                           +" << std::endl;
    std::cout << "+                  by thelleo                    +" << std::endl;
    std::cout << "+------------------------------------------------+" << std::endl;
    std::cout << "+ Variables naming:                              +" << std::endl;
    std::cout << "+ { a1*x1 + a2*x2 + a3*x3 = a4                   +" << std::endl;
    std::cout << "+ { a5*x1 + a6*x2 + a7*x3 = a8                   +" << std::endl;
    std::cout << "+ { a9*x1 + a10*x2 + a11*x3 = a12                +" << std::endl << std::endl;
    std::cout << "+ Transform your system to this                  +" << std::endl;
    std::cout << "+ form before typing values.                     +" << std::endl;
    std::cout << "+------------------------------------------------+" << std::endl << std::endl;

    unsigned long size;
    std::cout << "How many x-es (or rows) your system have? ";
    std::cin >> size;
    while(size < 2) {
        std::cout << "Positive integer > 1 please: ";
        std::cin >> size;
    }

    std::vector<std::vector<double>> coefficients;
    int index = 1;
    for(int i = 0; i < size; ++i) {
        std::vector<double> row;
        for(int j = 0; j < size + 1; ++j) {
            double input;
            std::cout << "Value of a" << index++ << ": ";
            std::cin >> input;
            row.push_back(input);
        }
        coefficients.push_back(row);
    }

    std::cout << std::endl << "Calculating..." << std::endl;
    CramerTheorem calculator(size, coefficients);
    calculator.calculate();
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
            break;
    }
    std::cout << std::endl;

    return 0;
}