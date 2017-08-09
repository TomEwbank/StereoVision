//
// Created by dallas on 09.08.17.
//

#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;


int main(int argc, char** argv) {
    try {

        String folderName = "imgs_rectified/";
        String calibName = "stereoCalib_2305_rotx008_nothingInv";//"stereoParams_2906";//
        String calibFile = folderName+calibName;
        bool fullHoropter = false;

        if (argc > 1 && !strcmp(argv[1], "-fh"))
            fullHoropter = true;

        FileStorage fs;
        fs.open(calibFile, FileStorage::READ);
        Rect commonROI;
        fs["common_ROI"] >> commonROI;
        Mat Q;
        fs["Q"] >> Q;

        if (fullHoropter) {

            int red[] = {120,
                         90,
                         70,
                         0,
                         0,
                         0,
                         195,
                         255,
                         255,
                         255};
            int green[] = {  0,
                             0,
                             0,
                             170,
                             200,
                             200,
                             255,
                             255,
                             155,
                             0  };
            int blue[] = {136,
                          184,
                          245,
                          225,
                          200,
                          125,
                          0,
                          0,
                          0,
                          0};

            // Generate pointCloud of horopter
            ofstream outputFile("disp_discretezation_cloud" + calibName + ".txt");
            for (int d = 1; d <= 200; ++d) {

                std::vector<Vec3d> vin(commonROI.width * commonROI.height);

                std::vector<Vec3d>::iterator vinIt = vin.begin();
                for (int i = commonROI.x; i < commonROI.x + commonROI.width; i += 20) {
                    for (int j = commonROI.y; j < commonROI.y + commonROI.height; j += 20) {
                        Vec3d p(i, j, d);
                        *vinIt = p;
                        ++vinIt;
                    }
                }

                std::vector<Vec3d> vout(vin.size());
                perspectiveTransform(vin, vout, Q);

                for (Vec3d point3D : vout) {

                    double x = point3D.val[0];
                    double y = point3D.val[1];
                    double z = point3D.val[2];

                    double r = red[d % 10];
                    double g = green[d % 10];
                    double b = blue[d % 10];

//                if (z > 0 && sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)) < 8000)
                    outputFile << x << " " << y << " " << z << " " << r << " " << g << " " << b << endl;
                }

            }
            outputFile.close();
        }


        std::vector<Vec3d> vin2(200);
        std::vector<Vec3d>::iterator vin2It = vin2.begin();
        for (int d = 1; d <= 200; ++d) {
            Vec3d p(commonROI.x + commonROI.width/2, commonROI.y + commonROI.height/2, d);
            *vin2It = p;
            ++vin2It;
        }

        std::vector<Vec3d> vout2(vin2.size());
        perspectiveTransform(vin2, vout2, Q);

        ofstream outputFile2("disp_discretezation_" + calibName + ".txt");
        vin2It = vin2.begin();
        for (Vec3d point3D : vout2) {

            double z = point3D.val[2];
            int d = (*vin2It).val[2];
            ++vin2It;

            outputFile2 << d << " " << z << endl;
        }
        outputFile2.close();


        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal exception: " << e.what();
        return 1;
    }
}



