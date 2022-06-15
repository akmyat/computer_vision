#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define IMAGE_FILE "../../../../Data/sample.jpg"

int main() {
    int key = -1;

    Mat image = imread(IMAGE_FILE);
    if(image.empty())
    {
        cout << "Error: No image to show" << endl;
        return 1;
    }

    imshow("Input image", image);

    // Wait up to 5s for a key press
    key = waitKey(5000);
    cout << "Key: " << key << endl;

    return 0;
}
