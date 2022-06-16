#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define VIDEO_FILE "../../../../Data/robot.mp4"
#define ROTATE false

int main() {
    Mat matFrameCapture;
    Mat matFrameDisplay;
    int currentFrameNum;
    int totalFrameNum;

    int key = int(' ');

    // Open video file
    VideoCapture videoCapture(VIDEO_FILE);
    double height = videoCapture.get(CAP_PROP_FRAME_HEIGHT);
    double width = videoCapture.get(CAP_PROP_FRAME_WIDTH);
    totalFrameNum = int(videoCapture.get(CAP_PROP_FRAME_COUNT));
    if(!videoCapture.isOpened())
    {
        cerr << "ERROR! Unable to open video file " << VIDEO_FILE << endl;
        return -1;
    }

    // Capture loop
    while(true)
    {
        // Get the next frame when press <space>
        if(key == int(' '))
        {
            videoCapture.read(matFrameCapture);
            currentFrameNum = int(videoCapture.get(CAP_PROP_POS_FRAMES));
            if(matFrameCapture.empty())
            {
                // End of video file
                break;
            }

            // Rotate video
            #if ROTATE
                        //Rotate 180 degree and put image to matFrameDisplay
                        rotate(matFrameCapture, matFrameDisplay, RotateFlags::ROTATE_180);
            #else
                        matFrameDisplay = matFrameCapture;
            #endif

            // Resize the frame
            double height_ratio = 768.0 / height;
            double width_ratio = 1366.0 / width;
            int down_height = int(height * height_ratio);
            int down_width = int(width * width_ratio);
            resize(matFrameDisplay,
                   matFrameDisplay,
                   Size(down_width, down_height),
                   width_ratio, height_ratio, INTER_LINEAR);

            // 1366p x 768p frame display in resizeable, keep aspect ratio, and show expended GUI
            namedWindow("ROBOT.MP4", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);

            // Display the frame
            imshow("ROBOT.MP4", matFrameDisplay);

            // Display overlay explanatory information
            string explanatory_info = std::to_string(currentFrameNum) + " / "
                                      + std::to_string(totalFrameNum)+" frames. Press <space> for next frame. Press <q> to quit.";
            displayOverlay("ROBOT.MP4", explanatory_info);
        }

        // Quit when press <q>
        if (key == int('q')) break;

        // Exit program on window close
        if (getWindowProperty("ROBOT.MP4", WND_PROP_VISIBLE) != 1) break;

        key = pollKey();
    }

    destroyAllWindows();
    return 0;
}
