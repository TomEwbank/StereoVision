//
// Created by dallas on 07.03.17.
//
#include <opencv2/opencv.hpp>
#include <vector>

#ifndef PROJECT_CENSUSTRANSFORM_H
#define PROJECT_CENSUSTRANSFORM_H

using namespace std;
using namespace cv;

class CensusTransform {
public:
    CensusTransform(Mat_<char> img, size_t size);
    vector<unsigned int> getCensus(int u, int v);
    size_t getWidth() {return width;};
    size_t getHeight() {return height;};
    size_t getNWords() {return nWords;};
    size_t getMaskSize() {return maskSize;};


private:
    vector<vector<unsigned int>> matrix;
    size_t width;
    size_t height;
    size_t nWords;
    size_t maskSize;

    vector<unsigned int> computeCensus(int u, int v, Mat_<char> img);
};


#endif //PROJECT_CENSUSTRANSFORM_H
