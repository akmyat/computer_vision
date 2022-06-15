#include <iostream>
#include <opencv2/core.hpp>
#include "matplotlib-cpp.h"

using namespace cv;
using namespace std;
namespace plt = matplotlibcpp;

int main()
{
    double P[12] = {2, 4, 2, 6, 3, 3, 1, 2, 0.5, 16, 8, 4};
    Mat matP(4, 3, CV_64F, P);

    double L[3] = {8, -4, 0};
    Mat matL(3, 1, CV_64F, L);

    Mat D = matP * matL;
    cout << "P: " << matP << endl << endl;
    cout << "L: " << matL << endl << endl;
    cout << "P . L = " << D << endl << endl;

    cout << "The following points are on line (8, -4, 0)" << endl;
    for (int i = 0; i<4; i++)
    {
        if(D.at<double>(0, i) == 0)
        {
            double x = matP.at<double>(i, 0) / matP.at<double>(i, 2);
            double y = matP.at<double>(i, 1) / matP.at<double>(i, 2);
            cout << "( " << x << ", " << y << " )" << endl;
        }
    }

    std::vector<double> X(4), Y(4), lineX(100), lineY(100);
    for (int i =0; i<4; i++)
    {
        X[i] = matP.at<double>(i, 0) / matP.at<double>(i, 2);
        Y[i] = matP.at<double>(i, 1) / matP.at<double>(i, 2);
    }

    double value = -0.04;
    double a = -matL.at<double>(0,0);
    double b = matL.at<double>(1, 0);
    double c = matL.at<double>(2, 0);
    cout << a << b << c << endl;
    for(int j = 0; j<100; j++)
    {
        value += 0.04;
        lineX[j] = value;
        lineY[j] = ((a * lineX[j]) - c) / b;
    }

    plt::plot(X, Y, "r*");
    plt::plot(lineX, lineY, "b-");
    plt::xlabel("X-axis");
    plt::ylabel("Y-axis");
    plt::title("2D line (8, -4, 0) and Four Homogeneous 2D points");
    plt::show();
}