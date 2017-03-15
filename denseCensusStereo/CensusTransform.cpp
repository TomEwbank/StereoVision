//
// Created by dallas on 07.03.17.
//

#include "CensusTransform.h"

// TODO: remove degug cout

CensusTransform::CensusTransform(Mat_<char> img, size_t size) {
    if (size%2 == 0)
        maskSize = size+1;
    else
        maskSize = size;

    width = img.cols;
    height = img.rows;
    nWords = ((maskSize*maskSize)/32)+1;

    for (int j=0; j<height; ++j){
        for (int i=0; i<width; ++i){
            //cout << j*width+i << endl;
            vector<unsigned int> census = this->computeCensus(i,j,img);
            matrix.push_back(census);
        }
    }

}

vector<unsigned int> CensusTransform::getCensus(int u, int v) {
    return matrix[v * width + u];
}

vector<unsigned int> CensusTransform::computeCensus(int u, int v, Mat_<char> img) {
    vector<unsigned int> result(nWords,0);

    if (u < (int)(maskSize/2) || v < (int)(maskSize/2) || u >= (int)(width-(maskSize/2)) || v >= (int)(height-(maskSize/2)))
        return result;

    char ref = img.ptr<char>(v)[u];
    int pass = 0;

    //cout << "ref  " << (int)ref << endl;
    for (int j=0; j<maskSize; ++j) {
        char* pixel = img.ptr<char>(v-(maskSize/2)+j);
        for (int i=0; i<maskSize; ++i) {
            //cout << j << " - " << i << endl;
            if(j==v && i==u) {
                //cout << "coucou" << endl;
                pass = -1;
                continue;
            }

            //cout << "pixel  " << (int)pixel[u-(maskSize/2)+i] << endl;
            //cout << "!! " << j*maskSize+i << " - " << j*maskSize+i+pass << endl;

            if(pixel[u-(maskSize/2)+i] >= ref) {
                int bit = j*maskSize+i+pass;
                //cout << "bit " << bit << endl;
                int n = bit/32;
                result[n] |= 1 << (bit%32);
                //cout << "temp " << result[n] << endl;
            }
        }
    }

    return result;
}