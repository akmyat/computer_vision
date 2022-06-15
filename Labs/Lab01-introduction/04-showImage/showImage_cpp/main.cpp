#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    string img = "/mnt/ntfs/Data/code/CV/CV/Labs/02-Homographies/img/sample.jpg";

    Mat srcImage = imread(img);
    if (!srcImage.data)
    {
        return 1;
    }
    imshow("srcImage", srcImage);
    waitKey(0);

    Mat greyMat;
    cvtColor(srcImage, greyMat, COLOR_BGR2GRAY);
    imshow("greyImage", greyMat);
    waitKey(0);

    return 0;
}
