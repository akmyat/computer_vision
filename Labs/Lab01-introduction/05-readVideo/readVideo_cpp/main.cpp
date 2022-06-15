#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#define VIDEO_FILE "../../../../data/robot.mp4"

using namespace std;
using namespace cv;

int main() {

    Mat matFrameCapture;

    int key = -1;
    
    // Open input video file
    VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened())
    {
        cerr << "Error. Unable to open input video file." << VIDEO_FILE << endl;
        return -1;
    }

    while (key < 0)
    {
        // Get next frame
        videoCapture.read(matFrameCapture);

        if(matFrameCapture.empty())
        {
            // End of video file
            break;
        }

        // Resize the frame
        double ratio = 480.0 / matFrameCapture.rows;
        resize(matFrameCapture, matFrameCapture, Size(), ratio, ratio, INTER_LINEAR);

        // Show the frame
        imshow(VIDEO_FILE, matFrameCapture);

        // Get key input
        key = waitKey(30);
    }
    return 0;
}
