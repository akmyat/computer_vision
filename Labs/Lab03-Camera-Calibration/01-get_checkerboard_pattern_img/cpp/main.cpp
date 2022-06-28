#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    int frameAdd = 0;
    Mat frame;
    int iKey = -1;

    // --- INITIALIZE VIDEO CAPTURE ---
    VideoCapture videoCapture;

    // --- Open the default camera using default API ---
    // videoCapture.open(0);


    // --- Open selected camera using selected API ---
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    videoCapture.open(deviceID, apiID);

    if (!videoCapture.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // --- Grab and Write Loop ---
    cout << "Start grabbing" << endl << "Press s to save images and q to terminate" << endl;

    while (true)
    {
        // Wait for a new frame from camera and store it into 'frame'
        videoCapture.read(frame);

        // Check frame is not empty
        if (frame.empty())
        {
            cerr << "ERROR! Blank frame grabbed\n";
            break;
        }

        // Show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);

        iKey = waitKey(5);
        if (iKey == 's' || iKey == 'S')
        {
            imwrite("../images/frame" + to_string(frameAdd) + ".jpg", frame);
            frameAdd++;
            cout << "Frame: " << frameAdd << " has been saved." << endl;
        }
        else if (iKey == 'q' || iKey == 'Q')
        {
            break;
        }

    }
    return 0;
}
