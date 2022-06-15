#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define VIDEO_FILE "../../../../Data/robot.mp4"
#define ROTATE false

int main() {
    Mat matFrameCapture;
    Mat matFrameDisplay;

    int key = -1;

    // Open video file
    VideoCapture videoCapture(VIDEO_FILE);
    double height = videoCapture.get(CAP_PROP_FRAME_HEIGHT);
    double width = videoCapture.get(CAP_PROP_FRAME_WIDTH);

    if(!videoCapture.isOpened())
    {
        cerr << "ERROR! Unable to open video file." << VIDEO_FILE << endl;
        return -1;
    }

    // Capture loop
    while (key != int(' '))
    {
        // Get the next frame
        videoCapture.read(matFrameCapture);
        if(matFrameCapture.empty())
        {
            // End of video file
            break;
        }

        // Rotate Video
        #if ROTATE
            // Roate 180 degree and put image to matFrameDisplay
            rotate(matFrameCapture, matFrameDisplay, RotateFlags::ROTATE_180);
        #else
            matFrameDisplay = matFrameCapture;
        #endif

        // Resize the frame
        double ratio = 480.0 / height;
        int down_height = int(height * ratio);
        int down_width = int(width * ratio);
        resize(matFrameDisplay,
               matFrameDisplay,
               Size(down_width, down_height),
               ratio, ratio, INTER_LINEAR);

        // Display the frame
        imshow(VIDEO_FILE, matFrameDisplay);

        key = waitKey(30);
    }
    return 0;
}
