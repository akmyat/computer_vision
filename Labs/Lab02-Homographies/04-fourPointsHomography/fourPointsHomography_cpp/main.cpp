#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#define VIDEO_FILE "../../../../Data/test.mp4"
#define ROTATE false

struct State
{
    Mat matPauseScreen, matResult, matFinal;
    Point point;
    vector<Point> pts;
    int var = 0;
    int drag = 0;
};

Mat homography_matrix;

void mouseHandler(int, int, int, int, void*);
void calcHomography(struct State *pState);

int main() {
    Mat matFrameCapture;
    Mat matFrameDisplay;
    int currentFrameNum;
    int totalFrameNum;

    int key = -1;

    struct State state;
    state.var = 0;
    state.drag = 0;

    // --------------------- [STEP 1: Make a video capture from file] ---------------------
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
    while(key < 0)
    {
        //Get the next frame
        videoCapture.read(matFrameCapture);
        currentFrameNum = int(videoCapture.get(CAP_PROP_POS_FRAMES));
        if (matFrameCapture.empty())
        {
            // End of video file
            break;
        }

        // Convert to Grayscale
//        cvtColor(matFrameCapture, matFrameCapture, COLOR_BGR2GRAY);

#if ROTATE
        // Rotate 180 degree and put image to matFrameCapture
        rotate(matFrameCapture, matFrameCapture, RotateFlags::ROTATE_180);
#endif

        // Resize the frame
        double height_ratio = 768.0 / height;
        double width_ratio = 1366.0 / width;
        int down_height = int(height * height_ratio);
        int down_width = int(width * height_ratio);
        resize(matFrameCapture, matFrameDisplay,
               Size(down_width, down_height),
               width_ratio, height_ratio, INTER_LINEAR);

        // Display the frame
        imshow("ROBOT.MP4", matFrameDisplay);

        // Display overlay explanatory information
        string explanatory_info = std::to_string(currentFrameNum) + "/"
                + std::to_string(totalFrameNum)
                + "frames. Press any key to Pause.";
        displayOverlay("ROBOT.MP4", explanatory_info);

        key = waitKey(30);

        if (key >= 0)
        {
            state.matPauseScreen = matFrameDisplay;
            state.matFinal = state.matPauseScreen.clone();
        }
    }
    if (!matFrameCapture.empty())
    {
        state.var = 0;  // reset number of saving points
        state.pts.clear();  // reset all points
        namedWindow("Source", WINDOW_AUTOSIZE); // Create window named Source
        setMouseCallback("Source", mouseHandler, &state);
        imshow("Source", state.matPauseScreen);
        waitKey(0);

        destroyWindow("Source");

        if (state.pts.size() == 4)
        {

            calcHomography(&state);

            cout << "Homography Matrix is: " << endl;
            cout << homography_matrix << endl;

            int rows = state.matPauseScreen.rows;
            int cols = state.matPauseScreen.cols;
            Mat dst = Mat::zeros(rows, cols, CV_32F);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
//                    int point[] = {i, j, 1};
//                    Mat tempMat(3,1, CV_32F, point);
//                    Mat res = homography_matrix * tempMat;
//                    int i2 = res.at<int>(0,0) / res.at<int>(2, 0) + 0.5;
//                    int j2 = res.at<int>(1, 0) / res.at<int>(2, 0) + 0.5;
//                    cout << i2 << " " << j2 << endl;
//                    if (i2 >= 0 && i2 < rows)
//                    {
//                        if(j2 >=0 && j2 < cols)
//                        {
//                            dst.at<float>(i2, j2) = state.matPauseScreen.at<float>(i, j);
//
//                        }
//                    }
                }
            }
            state.matResult = dst;
            cout << state.matResult.size() << endl;
//            warpPerspective(state.matPauseScreen,
//                            state.matResult,
//                            homography_matrix,
//                            state.matPauseScreen.size(),
//                            INTER_LINEAR );

            imshow("Source", state.matPauseScreen);
            imshow("Result", state.matResult);
            waitKey(0);
        }
    }
    else
    {
        cout << "You did not pause the screen before the video finish. The program will stop" << endl;
    }
    return 0;
}

void mouseHandler(int event, int x, int y, int, void *pVoid)
{
    auto *pState = (struct State *) pVoid;

    if (pState->var >= 4)    // If we already have 4 points, do nothing
        return;

    if (event == EVENT_LBUTTONDOWN) // Mouse Left Click
    {
        pState->drag = 1;   // Set it that the mouse is in pressing down mode
        pState->matResult = pState->matFinal.clone();   // copy final image to draw image
        pState->point = Point(x, y);
        if (pState->var >= 1)   // If more than 1 points has been added, draw a line
        {
            line(pState->matResult, pState->pts[pState->var - 1], pState->point, Scalar(0, 255, 0, 255), 2);
        }

        circle(pState->matResult, pState->point, 2, Scalar(0, 255, 0), -1, 8, 0);   // draw a current green point
        imshow("Source", pState->matResult);    // Show the current drawing
    }

    if (event == EVENT_LBUTTONUP && pState->drag)   // Mouse Left click release
    {
        pState->drag = 0;   // No more mouse drag
        pState->pts.push_back(pState->point);
        pState->var++;
        pState->matFinal = pState->matResult.clone();   // copy the current drawing image to final image
        if (pState->var >= 4)   // If the homography points are done
        {
            line(pState->matFinal, pState->pts[0], pState->pts[3], Scalar(0, 255, 0, 255), 2);  // draw the last line
            fillPoly(pState->matFinal, pState->pts, Scalar(0, 120, 0, 20), 8, 0);   // draw polygon from points
            setMouseCallback("Source", nullptr, nullptr); // remove mouse event handler
        }
        imshow("Source", pState->matFinal);
    }

    if (pState->drag)   // if the mouse is dragging
    {
        pState->matResult = pState->matFinal.clone();   // Copy final images to draw image
        pState->point = Point(x, y);    // memorize current mouse position to point var
        if (pState->var >= 1)
        {
            line(pState->matResult, pState->pts[pState->var -1], pState->point, Scalar(0, 255, 0, 255), 2); //draw a green line with thickness 2
        }
        circle(pState->matResult, pState->point, 2, Scalar(0, 255, 0), -1, 8, 0);   // draw a current green point
        imshow("Source", pState->matResult);    // Show the current drawing
    }
}

void calcHomography(struct State *pState)
{
    cout << "Calculating homography..." << endl;

    for(int i = 0; i < pState->pts.size(); i++)
    {
        cout << "Point" << i << ": " << pState->pts[i] << endl;
    }

    if (pState->pts.size() != 4)
    {
        cout << "Four points are needed for a homography..." << endl;
        return;
    }

    Mat matA = Mat::zeros(8, 9, CV_32F);
    float xprimes[] = {400.0, 600.0, 600.0, 400.0};
    float yprimes[] = {400.0, 400.0, 600.0, 600.0};

    for (int i = 0; i < pState->pts.size(); i++)
    {
        auto x = float(pState->pts[i].x);
        auto y = float(pState->pts[i].y);

        auto xprime = float(xprimes[i]);
        auto yprime = float(yprimes[i]);

        matA.at<float>(i*2, 0) = -x;
        matA.at<float>(i*2, 1) = -y;
        matA.at<float>(i*2, 2) = -1;
        matA.at<float>(i*2, 3) = 0;
        matA.at<float>(i*2, 4) = 0;
        matA.at<float>(i*2, 5) = 0;
        matA.at<float>(i*2, 6) = x * xprime;
        matA.at<float>(i*2, 7) = y * xprime;
        matA.at<float>(i*2, 8) = xprime;

        matA.at<float>(i*2+1, 0) = 0;
        matA.at<float>(i*2+1, 1) = 0;
        matA.at<float>(i*2+1, 2) = 0;
        matA.at<float>(i*2+1, 3) = -x;
        matA.at<float>(i*2+1, 4) = -y;
        matA.at<float>(i*2+1, 5) = -1;
        matA.at<float>(i*2+1, 6) = x * yprime;
        matA.at<float>(i*2+1, 7) = y * yprime;
        matA.at<float>(i*2+1, 8) = yprime;
    }

    cout << "Matrix A: " << endl;
    cout << matA << endl;

    Mat matU, matW, matVt;
    SVDecomp(matA, matW, matU, matVt, 4);

    cout << "Matrix W: " << endl;
    cout << matW << endl;

    cout << "Matrix U: " << endl;
    cout << matU << endl;

    cout << "Matrix Vt: " << endl;
    cout << matVt << endl;

    Mat matH = Mat::zeros(3, 3, CV_32F);
    int k = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            matH.at<float>(i,j) = matVt.at<float>(8, k);
            k++;
        }
    }
    cout << "Matrix H: " << endl;
    cout << matH << endl;

    homography_matrix = matH;
}