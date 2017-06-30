//
// Created by dallas on 09.04.17.
//
#include "highPerfStereoLib.h"


float getInterpolatedDisparity(Fade_2D &dt, unordered_map<MeshTriangle, Plane> &planeTable, Mat_<float> disparities, int x,
                               int y);

void line2(Mat& img, const Point& start, const Point& end,
           const Scalar& c1, const Scalar& c2) {
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

void highPerfStereo(cv::Mat_<unsigned char> leftImg,
                    cv::Mat_<unsigned char> rightImg,
                    StereoParameters parameters,
                    Mat_<float> &disparityMap,
                    vector<Point> &highGradPoints) {


    if(leftImg.data == NULL || rightImg.data == NULL)
        throw sparsestereo::Exception("Input images are empty!");

    if(leftImg.cols != rightImg.cols || leftImg.rows != rightImg.rows)
        throw sparsestereo::Exception("Input images do not have the same dimensions!");

    if(leftImg.cols != disparityMap.cols || leftImg.rows != disparityMap.rows)
        throw sparsestereo::Exception("disparity map do not have the same dimensions as the input images!");


    // Stereo matching parameters
    //TODO decide if some parameters like nIter shouldnt be a particular argument of the function
    double uniqueness = parameters.uniqueness;
    int maxDisp = parameters.maxDisp;
    int leftRightStep = parameters.leftRightStep;
    int costAggrWindowSize = parameters.costAggrWindowSize;
    uchar gradThreshold = parameters.gradThreshold;
    char tLow = parameters.tLow;
    char tHigh = parameters.tHigh;
    int nIters = parameters.nIters;
    double resizeFactor = parameters.resizeFactor;
    bool applyBlur = parameters.applyBlur;
    bool applyHistEqualization = parameters.applyHistEqualization;
    int blurSize = parameters.applyBlur;

    // Feature detection parameters
    double adaptivity = parameters.adaptivity;
    int minThreshold = parameters.minThreshold;
    bool traceLines = parameters.traceLines;
    int nbLines = parameters.nbLines;
    int lineSize = parameters.lineSize;
    bool invertRows = parameters.invertRows;
    int nbRows = parameters.nbRows;

    // Gradient parameters
    int kernelSize = parameters.kernelSize;
    int scale = parameters.scale;
    int delta = parameters.delta;
    int ddepth = parameters.ddepth;

    // Misc. parameters
    bool recordFullDisp = parameters.recordFullDisp;
    bool showImages = parameters.showImages;
    int colorMapSliding = parameters.colorMapSliding;

    // Variables used if showImages true (not parameters!)
    int minDisparityFound = maxDisp;
    int maxDisparityFound = 0;


    // TODO check that it does not modify the original image outside the fnction
    leftImg = leftImg.clone();
    rightImg = rightImg.clone();

    // TODO deal with resize parameter
//    cv::Mat_<unsigned char> leftImg, rightImg;
//    resize(leftImgInit, leftImg, Size(), resizeFactor, resizeFactor);
//    resize(rightImgInit, rightImg, Size(), resizeFactor, resizeFactor);

    // Crop image so that SSE implementation won't crash
    //cv::Rect myROI(0,0,1232,1104);
    cv::Rect newROI(0,0,16*(leftImg.cols/16),16*(leftImg.rows/16));
    leftImg = leftImg(newROI);
    rightImg = rightImg(newROI);

    if (applyHistEqualization) {

        equalizeHist(leftImg, leftImg);
        equalizeHist(rightImg, rightImg);
    }

    // Apply Laplace function
    Mat grd, abs_grd;

    cv::Laplacian( leftImg, grd, ddepth, kernelSize, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grd, abs_grd );

    if (showImages) {
        // Show what you got
        namedWindow("Gradient left");
        imshow("Gradient left", abs_grd);
        waitKey(0);
    }

    // Get the set of high gradient points
    int v = 0;
    Mat highGradMask(grd.rows, grd.cols, CV_8U, Scalar(0));
    for(int j = 2; j < abs_grd.rows-2; ++j) {
        uchar* pixel = abs_grd.ptr(j);
        for (int i = 2; i < abs_grd.cols-2; ++i) {
            if (pixel[i] > gradThreshold) {
                highGradPoints.push_back(Point(i, j));
                highGradMask.at<uchar>(highGradPoints[v]) = (uchar) 200;
                ++v;
            }
        }
    }

    if (showImages) {
        // Show what you got
        namedWindow("Gradient mask");
        imshow("Gradient mask", highGradMask);
        waitKey(0);
    }

    if (applyBlur) {
        GaussianBlur(leftImg, leftImg, Size(blurSize, blurSize), 0, 0);
        GaussianBlur(rightImg, rightImg, Size(blurSize, blurSize), 0, 0);
    }


    // Horizontal lines tracing in images for better feature detection
    cv::Mat_<unsigned char> leftImgAltered, rightImgAltered;
    leftImgAltered = leftImg.clone();
    rightImgAltered = rightImg.clone();

    if (invertRows) {
        int rowSize = (leftImgAltered.rows / nbRows)+1;
        nbRows = leftImgAltered.rows/rowSize;
        for(int i = 0; i<(nbRows-1); ++i) {
            if (i%2 == 0) {
                Mat rowLeft = leftImgAltered(Rect(0,i*rowSize,leftImgAltered.cols,rowSize));
                Mat rowRight = rightImgAltered(Rect(0,i*rowSize,leftImgAltered.cols,rowSize));

                Mat_<unsigned char> inverter(rowSize, leftImgAltered.cols, 255);
                subtract(inverter, rowLeft, rowLeft);
                subtract(inverter, rowRight, rowRight);
            }
        }

        if ((nbRows-1)%2 == 0) {
            int lastRowSize = leftImgAltered.rows-rowSize*(nbRows-1);

            Mat rowLeft = leftImgAltered(Rect(0,(nbRows-1)*rowSize,leftImgAltered.cols,lastRowSize));
            Mat rowRight = rightImgAltered(Rect(0,(nbRows-1)*rowSize,leftImgAltered.cols,lastRowSize));

            Mat_<unsigned char> inverter(lastRowSize, leftImgAltered.cols, 255);
            subtract(inverter, rowLeft, rowLeft);
            subtract(inverter, rowRight, rowRight);
        }
    }


    if (traceLines) {
        int stepSize = leftImgAltered.rows / nbLines;
        for (int k = stepSize / 2; k < leftImgAltered.rows; k = k + stepSize) {
            for (int j = k; j < k + lineSize && j < leftImgAltered.rows; ++j) {
                leftImgAltered.row(j).setTo(0);
                rightImgAltered.row(j).setTo(0);
            }
        }
    }


    if (showImages) {
        // Show what you got
        namedWindow("left altered image");
        imshow("left altered image", leftImgAltered);
        waitKey(0);
        namedWindow("right altered image");
        imshow("right altered image", rightImgAltered);
        waitKey(0);
    }

    // Load rectification data
    StereoRectification* rectification = NULL;
//        if(calibFile != NULL)
//            rectification = new StereoRectification(CalibrationResult(calibFile));

    // The stereo matcher. SSE Optimized implementation is only available for a 5x5 window
    SparseStereo<CensusWindow<5>, short> stereo(maxDisp, 1, uniqueness,
                                                rectification, false, false, leftRightStep);

    // Feature detectors for left and right image
    FeatureDetector* leftFeatureDetector = new ExtendedFAST(true, minThreshold, adaptivity, false, 2);
    FeatureDetector* rightFeatureDetector = new ExtendedFAST(false, minThreshold, adaptivity, false, 2);

    vector<SparseMatch> correspondences;

    // Objects for storing final and intermediate results
    cv::Mat_<char> charLeft(leftImg.rows, leftImg.cols),
            charRight(rightImg.rows, rightImg.cols);
    Mat_<unsigned int> censusLeft(leftImg.rows, leftImg.cols),
            censusRight(rightImg.rows, rightImg.cols);
    vector<KeyPoint> keypointsLeft, keypointsRight;

    // Featuredetection. This part can be parallelized with OMP
#pragma omp parallel sections default(shared) num_threads(2)
    {
#pragma omp section
        {
            keypointsLeft.clear();
            leftFeatureDetector->detect(leftImgAltered, keypointsLeft);
            ImageConversion::unsignedToSigned(leftImg, &charLeft);
            Census::transform5x5(charLeft, &censusLeft);
        }
#pragma omp section
        {
            keypointsRight.clear();
            rightFeatureDetector->detect(rightImgAltered, keypointsRight);
            ImageConversion::unsignedToSigned(rightImg, &charRight);
            Census::transform5x5(charRight, &censusRight);
        }
    }

    // Stereo matching. Not parallelized (overhead too large)
    stereo.match(censusLeft, censusRight, keypointsLeft, keypointsRight, &correspondences);

    if (showImages) {

        // Find max and min disparity to adjust the color mapping of the depth so that the view will be better
        minDisparityFound = maxDisp;
        maxDisparityFound = 0;
        for (std::vector<sparsestereo::SparseMatch>::const_iterator it = correspondences.begin();
             it < correspondences.end();
             it++) {

            int disp = it->disparity();
            if(disp < minDisparityFound)
                minDisparityFound = disp;
            if(disp > maxDisparityFound)
                maxDisparityFound = disp;
        }


        // Highlight matches as colored boxes
        Mat_<Vec3b> screen(leftImg.rows, leftImg.cols);
        cvtColor(leftImg, screen, CV_GRAY2BGR);

        for (int i = 0; i < (int) correspondences.size(); i++) {

            // Generate the color associated to the disparity value
            double scaledDisp = (double) (correspondences[i].disparity()-minDisparityFound) / (maxDisparityFound-minDisparityFound);
            Vec3b color = HLS2BGR((float) std::fmod(scaledDisp * 359+colorMapSliding, 360), 0.5, 1);

            // Draw the small colored box
            rectangle(screen, correspondences[i].imgLeft->pt - Point2f(2, 2),
                      correspondences[i].imgLeft->pt + Point2f(2, 2),
                      (Scalar) color, CV_FILLED);
        }


        // Display image and wait
        namedWindow("Sparse stereo");
        imshow("Sparse stereo", screen);
        waitKey();
    }


    // Create the triangulation mesh
    Fade_2D dt;
    Mat_<float> disparities(leftImg.rows, leftImg.cols, (float) 0); // Holds the temporary disparities
    for(int i=0; i<(int)correspondences.size(); i++) {
        float x = correspondences[i].imgLeft->pt.x;
        float y = correspondences[i].imgLeft->pt.y;
        float d = correspondences[i].disparity();

        disparities.at<float>(Point(x,y)) = d;

        Point2 p(x, y);
        dt.insert(p);
    }

    // Init final cost map
    Mat_<char> finalCosts(leftImg.rows, leftImg.cols, (char) 25);
    unsigned int occGridSize = 64;

    for (int iter = 1; iter <= nIters; ++iter) {

        // Fill lookup table for plane parameters
        unordered_map<MeshTriangle, Plane> planeTable;
        std::set<std::pair<Point2 *, Point2 *> > sEdges;
        std::vector<Triangle2 *> vAllTriangles;
        dt.getTrianglePointers(vAllTriangles);
        for (std::vector<Triangle2 *>::iterator it = vAllTriangles.begin();
             it != vAllTriangles.end(); ++it) {

            MeshTriangle mt = {*it};
            unordered_map<MeshTriangle, Plane>::const_iterator got = planeTable.find(mt);
            if (got == planeTable.end()) {
                Plane plane = Plane(*it, disparities);
                planeTable[mt] = plane;
            }
            // TODO optimize by not recompute plane parameters for unchanged triangles

            if (showImages) {
                // Use this loop over the triangles to retrieve all unique edges to display
                for (int i = 0; i < 3; ++i) {
                    Point2 *p0((*it)->getCorner((i + 1) % 3));
                    Point2 *p1((*it)->getCorner((i + 2) % 3));
                    if (p0 > p1) std::swap(p0, p1);
                    sEdges.insert(std::make_pair(p0, p1));
                }
            }
        }

        if (showImages) {

            // Find max and min disparity to adjust the color mapping of the depth so that the view will be better
            minDisparityFound = maxDisp;
            maxDisparityFound = 0;
            std::vector<GEOM_FADE2D::Point2 *> vAllPoints;
            dt.getVertexPointers(vAllPoints);

            for (std::vector<GEOM_FADE2D::Point2 *>::const_iterator it = vAllPoints.begin();
                 it < vAllPoints.end();
                 it++) {

                Point2 *p = *it;
                int disp = disparities.at<float>(Point(p->x(), p->y()));

                if(disp < minDisparityFound)
                    minDisparityFound = disp;
                if(disp > maxDisparityFound)
                    maxDisparityFound = disp;
            }

            // Display mesh
            Mat_<Vec3b> mesh(leftImg.rows, leftImg.cols);
            cvtColor(leftImg, mesh, CV_GRAY2BGR);
            set<std::pair<Point2 *, Point2 *>>::const_iterator pos;

            for (pos = sEdges.begin(); pos != sEdges.end(); ++pos) {

                Point2 *p1 = pos->first;
                float scaledDisp = (disparities.at<float>(Point(p1->x(), p1->y()))-minDisparityFound)
                                   / (maxDisparityFound-minDisparityFound);
                Vec3b color1 = HLS2BGR((float) std::fmod(scaledDisp * 359+colorMapSliding, 360), 0.5, 1);

                Point2 *p2 = pos->second;
                scaledDisp = (disparities.at<float>(Point(p2->x(), p2->y()))-minDisparityFound)
                             / (maxDisparityFound-minDisparityFound);
                Vec3b color2 = HLS2BGR((float) std::fmod(scaledDisp * 359+colorMapSliding, 360), 0.5, 1);


                line2(mesh, Point(p1->x(), p1->y()), Point(p2->x(), p2->y()), (Scalar) color1, (Scalar) color2);
            }


            // Display image and wait
            namedWindow("Triangular mesh");
            imshow("Triangular mesh", mesh);
            waitKey();
        }


        // TODO interpolate disparities for high grad points only, even if no showing images
        // Disparity interpolation
        if (showImages || recordFullDisp) {
            // interpolate on the complete image to be able to display the dense disparity map, or record it
            for (int j = 0; j < disparities.rows; ++j) {
                float *pixel = disparities.ptr<float>(j);
                for (int i = 0; i < disparities.cols; ++i) {
                    float disparity = getInterpolatedDisparity(dt, planeTable, disparities, i, j);
                    if(disparity >= 0)
                        pixel[i] = disparity;
                }
            }

            // Display interpolated disparities
            cv::Mat dst = disparities / maxDisp;
            if (showImages) {

                namedWindow("Full interpolated disparities");
                imshow("Full interpolated disparities", dst);
                waitKey();
            }

            if (recordFullDisp) {
                Mat outputImg;
                Mat temp = dst * 255;
                temp.convertTo(outputImg, CV_8UC1);
                imwrite("disparity" + to_string(iter) + ".png", outputImg);
            }
        } else {
            // No need to show dense disparity map -> interpolate disparities for high gradient points only
            for(Point &p : highGradPoints) {
                int i = p.x;
                int j = p.y;
                float disparity = getInterpolatedDisparity(dt, planeTable, disparities, i, j);
                if(disparity >= 0) {
                    float *pixel = disparities.ptr<float>(j);
                    pixel[i] = disparity;
                }
            }
        }

        Mat_<char> matchingCosts(leftImg.rows, leftImg.cols, tHigh);
        costEvaluation(censusLeft, censusRight, highGradPoints, disparities, matchingCosts);
        PotentialSupports ps = disparityRefinement(highGradPoints, disparities, matchingCosts,
                                                   tLow, tHigh, occGridSize, disparityMap, finalCosts);

        if(showImages) {

            // Highlight matches as colored boxes
            Mat_<Vec3b> badPts(leftImg.rows, leftImg.cols);
            cvtColor(leftImg, badPts, CV_GRAY2BGR);

            for (unsigned int i = 0; i < ps.getOccGridWidth(); ++i) {
                for (unsigned int j = 0; j < ps.getOccGridHeight(); ++j) {
                    InvalidMatch p = ps.getInvalidMatch(i, j);
                    rectangle(badPts, Point2f(p.x, p.y) - Point2f(2, 2),
                              Point2f(p.x, p.y) + Point2f(2, 2),
                              (Scalar) Vec3b(0, 255, 0), CV_FILLED);
                }
            }

            namedWindow("Candidates for epipolar matching");
            imshow("Candidates for epipolar matching", badPts);
            waitKey();

            cv::Mat dst = (disparityMap-minDisparityFound) / (maxDisparityFound-minDisparityFound);

            Mat finalColorDisp(disparityMap.rows, disparityMap.cols, CV_8UC3, Scalar(0, 0, 0));
            for (int y = 0; y < finalColorDisp.rows; ++y) {
                Vec3b *colorPixel = finalColorDisp.ptr<Vec3b>(y);
                float *pixel = dst.ptr<float>(y);
                for (int x = 0; x < finalColorDisp.cols; ++x)
                    if (pixel[x] > 0) {
                        Vec3b color = HLS2BGR((float) std::fmod(pixel[x] * 359+colorMapSliding, 360), 0.5, 1);
                        colorPixel[x] = color;
                    }
            }


            namedWindow("High gradient color disparities");
            imshow("High gradient color disparities", finalColorDisp);
            waitKey();
        }

        if (iter != nIters) {

            // Support resampling
            cout << "bitch" << endl;
            supportResampling(dt, ps, censusLeft, censusRight, 5, costAggrWindowSize, disparities, tLow, tHigh, maxDisp);
            occGridSize = max((unsigned int) 1, occGridSize / 2);
        }
    }

    // Add all support points to disparities
    std::vector<GEOM_FADE2D::Point2 *> vAllPoints;
    dt.getVertexPointers(vAllPoints);

    for (std::vector<GEOM_FADE2D::Point2 *>::const_iterator it = vAllPoints.begin();
         it < vAllPoints.end();
         it++) {

        Point2 *p = *it;
        Point pixel(p->x(), p->y());
        int disp = disparities.at<float>(pixel);
        disparityMap.at<float>(pixel) = disp;
        highGradPoints.push_back(pixel);
    }

    // Clean up
    delete leftFeatureDetector;
    delete rightFeatureDetector;
    if(rectification != NULL)
        delete rectification;

}

float getInterpolatedDisparity(Fade_2D &dt,
                               unordered_map<MeshTriangle, Plane> &planeTable,
                               Mat_<float> disparities,
                               int x, int y) {

    // Express the point for use with the Fade library
    Point2 pointInPlaneFade = Point2(x,y);

    // Find the triangle to which the point belong
    Triangle2 *t = dt.locate(pointInPlaneFade);
    MeshTriangle mt = {t};

    if (t != NULL) {
        // Look for the triangle in the lookup table
        unordered_map<MeshTriangle, Plane>::const_iterator got = planeTable.find(mt);
        Plane plane;

        if (got == planeTable.end()) {
            // If the triangle is not yet in the lookup table,
            // create a Plane object add it to the table
            plane = Plane(t, disparities);
            planeTable[mt] = plane;
        } else {
            // If the triangle is already in the lookup table, get the corresponding Plane object
            plane = got->second;
        }
        // Set the disparity value, obtained from the Plane object at the considered point
        return plane.getDepth(pointInPlaneFade);
    } else {
        return -1;
    }

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
