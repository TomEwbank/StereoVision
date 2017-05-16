//
// Created by dallas on 09.04.17.
//
#include "highPerfStereoLib.h"


void line2(Mat& img, const Point& start, const Point& end,
           const Scalar& c1,   const Scalar& c2) {
    LineIterator iter(img, start, end, 8);

    for (int i = 0; i < iter.count; i++, iter++) {
        double alpha = double(i) / iter.count;
        // note: using img.at<T>(iter.pos()) is faster, but
        // then you have to deal with mat type and channel number yourself
        img(Rect(iter.pos(), Size(1, 1))) = c1 * (1.0 - alpha) + c2 * alpha;
    }
}


void costEvaluation(const Mat_<unsigned int>& censusLeft,
                    const Mat_<unsigned int>& censusRight,
                    const vector<Point>& highGradPts,
                    const Mat_<float>& disparities,
                    Mat_<char>& matchingCosts) {

    HammingDistance h;
    vector<Point>::const_iterator i;
    for( i = highGradPts.begin(); i != highGradPts.end(); i++){
        float d = disparities.ptr<float>(i->y)[i->x];
        int xRight = i->x-(int)floor(d+0.5);

        if (xRight < 2)
            continue;
        else {
            unsigned int cl = censusLeft.ptr<unsigned int>(i->y)[i->x];
            unsigned int cr = censusRight.ptr<unsigned int>(i->y)[xRight];
            matchingCosts.ptr<char>(i->y)[i->x] = h.calculate(cl, cr);
        }
    }

}


PotentialSupports disparityRefinement(const vector<Point>& highGradPts,
                                      const Mat_<float>& disparities,
                                      const Mat_<char>& matchingCosts,
                                      const char tLow, const char tHigh,
                                      const unsigned int occGridSize,
                                      Mat_<float>& finalDisparities,
                                      Mat_<char>& finalCosts) {

    unsigned int occGridHeight = (unsigned int) (disparities.rows/occGridSize) + 1;
    unsigned int occGridWidth = (unsigned int) (disparities.cols/occGridSize) +1;
    // TODO: ensure no more bugs at limits of grid

    PotentialSupports ps(occGridHeight, occGridWidth, tLow, tHigh);

    vector<Point>::const_iterator it;
    for( it = highGradPts.begin(); it != highGradPts.end(); it++) {
        int u = it->x;
        int v = it->y;
        float d = disparities.ptr<float>(v)[u];
        char mc = matchingCosts.ptr<char>(v)[u];

        // Get occupancy grid indices for the considered point
        int i = u / occGridSize;
        int j = v / occGridSize;

        // If matching cost is lower than previous best final cost
        if (mc < finalCosts.ptr(v)[u]) {
            finalDisparities.ptr<float>(v)[u] = d;
            finalCosts.ptr<char>(v)[u] = mc;
        }

        // If matching cost is lower than previous best valid cost
        if (mc < tLow && mc < ps.getConfidentSupport(i,j).cost)
            ps.setConfidentSupport(i, j, u, v, d, mc);

        // If matching cost is higher than previous worst invalid cost
        if (mc > tHigh && mc > ps.getInvalidMatch(i,j).cost)
            ps.setInvalidMatch(i, j, u, v, mc);

    }

    return ps;
}

ConfidentSupport epipolarMatching(const Mat_<unsigned int>& censusLeft,
                                  const Mat_<unsigned int>& censusRight,
                                  int censusSize, int costAggrWindowSize,
                                  InvalidMatch leftPoint, int maxDisparity) {

//    const unsigned int *rightEpipolar = censusRight.ptr<unsigned int>(leftPoint.y);
//    HammingDistance h;
//    unsigned int censusRef = censusLeft.ptr<unsigned int>(leftPoint.y)[leftPoint.x];
//    int minCost = h.calculate(censusRef,rightEpipolar[leftPoint.x])+1;
//    int matchingX = leftPoint.x;
//    for(int i = leftPoint.x; i>=5 && i>(leftPoint.x-maxDisparity); --i) {
//        int cost = h.calculate(censusRef,rightEpipolar[i]);
//
//        if(cost < minCost) {
//            matchingX = i;
//            minCost = cost;
//        }
//    }
//
//    ConfidentSupport result = {leftPoint.x, leftPoint.y, (float)(leftPoint.x-matchingX), 0};
//    //cout  << ">> " << result.x << ", " << result.y << ", " << result.disparity << ", " << result.cost << endl;
//    return result;

    HammingDistance h;
    int minCost =  2147483647;//32*5*5;
    int matchingX = leftPoint.x;
    int halfWindowSize = costAggrWindowSize/2;
    int censusMargin = censusSize/2;
    for(int i = leftPoint.x; i>=halfWindowSize+censusMargin && i>(leftPoint.x-maxDisparity); --i) {
        int cost = 0;
        for (int m=-halfWindowSize; m<=halfWindowSize && leftPoint.y+m < censusLeft.rows; ++m) {

            if (leftPoint.y+m < 0)
                continue;

            const unsigned int* cl = censusLeft.ptr<unsigned int>(leftPoint.y+m);
            const unsigned int* cr = censusRight.ptr<unsigned int>(leftPoint.y+m);
            for (int n = -halfWindowSize; n <= halfWindowSize && i+n < censusLeft.cols; ++n) {
                cost += (int) h.calculate(cl[leftPoint.x+n], cr[i+n]);
            }
        }

        if(cost < minCost) {
            matchingX = i;
            minCost = cost;
        }
    }

    ConfidentSupport result(leftPoint.x, leftPoint.y, (float) (leftPoint.x-matchingX), h.calculate(censusLeft.ptr<unsigned int>(leftPoint.y)[leftPoint.x], censusRight.ptr<unsigned int>(leftPoint.y)[matchingX]));

    return result;
}

void supportResampling(Fade_2D &mesh,
                       PotentialSupports &ps,
                       const Mat_<unsigned int> &censusLeft,
                       const Mat_<unsigned int> &censusRight,
                       int censusSize, int costAggrWindowSize,
                       Mat_<float> &disparities,
                       char tLow, char tHigh, int maxDisp) {

    unsigned int occGridHeight = ps.getOccGridHeight();
    unsigned int occGridWidth = ps.getOccGridWidth();
    for (unsigned int j = 0; j < occGridHeight; ++j) {
        for (unsigned int i = 0; i < occGridWidth; ++i) {

            // sparse epipolar stereo matching for invalid pixels and add them to support points
            InvalidMatch invalid = ps.getInvalidMatch(i,j);
            if (invalid.cost > tHigh) {
                ConfidentSupport newSupp = epipolarMatching(censusLeft, censusRight, censusSize, costAggrWindowSize, invalid, maxDisp);
                if (newSupp.cost<tLow) {
                    disparities.ptr<float>(newSupp.y)[newSupp.x] = newSupp.disparity;
                    Point2 p(newSupp.x, newSupp.y);
                    mesh.insert(p);
                }
            }

            // add confident pixels to support points
            ConfidentSupport newSupp = ps.getConfidentSupport(i,j);
            //cout << newSupp.x << " " << newSupp.y << endl;
            if (newSupp.cost < tLow) {
                disparities.ptr<float>(newSupp.y)[newSupp.x] = newSupp.disparity;
                Point2 p(newSupp.x, newSupp.y);
                mesh.insert(p);
            }
        }
    }
}

cv::Vec3b ConvertColor( cv::Vec3b src, int code)
{
    cv::Mat srcMat(1, 1, CV_8UC3 );
    *srcMat.ptr< cv::Vec3b >( 0 ) = src;

    cv::Mat resMat;
    cv::cvtColor( srcMat, resMat, code);

    return *resMat.ptr< cv::Vec3b >( 0 );
}

cv::Vec3b HLS2BGR(float h, float l, float s) {
    if (h < 0 || h >= 360 || l < 0 || l > 1 || s < 0 || s > 1)
        throw invalid_argument("invalid HLS parameters");

    float c = (1 - abs(2*l-1))*s;
    float x = c*(1-abs(std::fmod(h/60,2)-1));
    float m = l - c/2;

    float r,g,b;

    if (h < 60) {
        r = c;
        g = x;
        b = 0;
    } else if (h < 120) {
        r = x;
        g = c;
        b = 0;
    } else if (h < 180) {
        r = 0;
        g = c;
        b = x;
    } else if (h < 240) {
        r = 0;
        g = x;
        b = c;
    } else if (h < 300) {
        r = x;
        g = 0;
        b = c;
    } else {
        r = c;
        g = 0;
        b = x;
    }

    r = (r+m)*255;
    g = (g+m)*255;
    b = (b+m)*255;

    Vec3b color((uchar)b,(uchar)g,(uchar)r);
    return color;
}
