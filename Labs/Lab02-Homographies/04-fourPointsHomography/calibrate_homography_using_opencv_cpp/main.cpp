#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

#define VIDEO_FILE "../../../../Data/robot.mp4"
#define CALIBRATION_FILE "robot_calibration.yml"
#define ROTATE false

struct State
{
    Mat matPauseScreen, matResult, matFinal;
    Point point;
    vector<Point> pts;
    int var = 0;
    int drag = 0;
};

void mouseHandler(int, int, int, int, void*);
void calibrate(struct State *pState);
void showSourceResultWithOpticalFlow();

int main() {
    struct State state;
    state.var = 0;
    state.drag = 0;

//    calibrate(&state);
    showSourceResultWithOpticalFlow();

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

void calibrate(struct State *pState)
{
    // --------------------- [STEP 1: Make video capture from file] ---------------------
    // Open video file
    VideoCapture videoCapture(VIDEO_FILE);

    if (!videoCapture.isOpened())
    {
        cout << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return;
    }

    double width = videoCapture.get(CAP_PROP_FRAME_WIDTH);
    double height = videoCapture.get(CAP_PROP_FRAME_HEIGHT);
    double ratio = 640.0 / width;
    int resizeHeight = int(height * ratio);
    int resizeWidth = int(width * ratio);

    int key = -1;
    Mat matFrameCapture;
    Mat matFrameDisplay;

    // Capture loop
    while (key < 0)
    {
        // Get the next frame
        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty())
        {
            break;
        }

        // Rotate
#if ROTATE
        rotate(matFrameCapture, matFrameDisplay, RotateFlags::ROTATE_180)
#else
        matFrameDisplay = matFrameCapture;
#endif
        // Resize
        resize(matFrameDisplay, matFrameDisplay, Size(resizeWidth, resizeHeight), ratio, ratio, INTER_LINEAR);

        imshow("ROBOT.MP4", matFrameDisplay);
        key = waitKey(30);

        //--------------------- [STEP 2: pause the screen and show an image] ---------------------
        if (key >= 0)
        {
            pState->matPauseScreen = matFrameCapture;
            pState->matFinal = pState->matPauseScreen.clone();
        }
    }

    // --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
    if (!matFrameCapture.empty())
    {
        pState->var = 0;
        pState->pts.clear();

        namedWindow("Source", WINDOW_NORMAL);
        setMouseCallback("Source", mouseHandler, pState);
        imshow("Source", pState->matPauseScreen);

        waitKey(0);
        destroyWindow("Source");

        if (pState->pts.size() == 4)
        {
            Point2f src[4];
            Point2f reals[4];

            src[0] = pState->pts[0];
            src[1] = pState->pts[1];
            src[2] = pState->pts[2];
            src[3] = pState->pts[3];

            reals[0] = Point2f(800, 800);
            reals[1] = Point2f(1000, 800);
            reals[2] = Point2f(1000, 1000);
            reals[3] = Point2f(800, 1000);

            for (int i = 0; i < 4; i++)
            {
                cout << "Capture " << i << ": " << src[i] << endl;
            }
            for (int i = 0; i < 4; i++)
            {
                cout << "Reals " << i << ": " << reals[i] << endl;
            }

            // --------------------- [STEP 4: Calculate Homography] ---------------------
            Mat homographyMatrix = getPerspectiveTransform(src, reals);
            cout << "\nEstimated Homography Matrix: " << endl;
            cout << homographyMatrix << endl;

            // --------------------- [STEP 5: Save Homography to file] ---------------------
            FileStorage cvFile(CALIBRATION_FILE, FileStorage::WRITE);
            cvFile.write("H", homographyMatrix);
            cvFile.release();

            // --------------------- [STEP 6: Warp Inverse Bi-linear Interpolation] ---------------------
            cv::Size s = pState->matPauseScreen.size();
            int h = s.height;
            int w = s.width;
            warpPerspective(pState->matPauseScreen, pState->matResult, homographyMatrix, Size(w, h), INTER_LINEAR);

            Mat matSourceFrame, matResultFrame;
            resize(pState->matPauseScreen, matSourceFrame, Size(resizeWidth, resizeHeight), ratio, ratio, INTER_LINEAR);
            resize(pState->matResult, matResultFrame, Size(resizeWidth, resizeHeight), ratio, ratio, INTER_LINEAR);

            imshow("Source", matSourceFrame);
            imshow("Result", matResultFrame);
            waitKey(0);
        }
        else
        {
            cout << "Required 4 point to calculate Homography!" << endl;
        }
    }
    else
    {
        cout << "No pause before end of video finish. Exiting." << endl;
    }
}

void showSourceResultWithOpticalFlow()
{
    FileStorage cvFile(CALIBRATION_FILE, FileStorage::READ);
    Mat matH;
    cvFile["H"] >> matH;
    cvFile.release();

    // Open video file
    VideoCapture videoCapture(VIDEO_FILE);

    if (!videoCapture.isOpened())
    {
        cout << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return;
    }

    double width = videoCapture.get(CAP_PROP_FRAME_WIDTH);
    double height = videoCapture.get(CAP_PROP_FRAME_HEIGHT);
    double ratio = 640.0 / width;
    int resizeHeight = int(height * ratio);
    int resizeWidth = int(width * ratio);

    Mat matFrameCapture;
    Mat matFrameDisplay;

    vector<Scalar>colors;
    RNG rng;
    for (int i=0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.emplace_back(r, g, b);
    }

    Mat old_frame_src, old_gray_src;
    Mat old_frame_res, old_gray_res;
    vector<Point2f> p0Src, p1Src;
    vector<Point2f> p0Res, p1Res;

    Mat frame;
    videoCapture.read(frame);
    old_frame_src = frame;
    warpPerspective(frame, old_frame_res, matH, frame.size(), INTER_LINEAR);

    cvtColor(old_frame_src, old_gray_src, COLOR_BGR2GRAY);
    cvtColor(old_frame_res, old_gray_res, COLOR_BGR2GRAY);

    goodFeaturesToTrack(old_gray_src, p0Src, 100, 0.3, 7, Mat(), 7, false, 0.04);
    goodFeaturesToTrack(old_gray_res, p0Res, 100, 0.3, 7, Mat(), 7, false, 0.04);

    Mat maskSrc = Mat::zeros(old_frame_src.size(), old_frame_src.type());
    Mat maskRes = Mat::zeros(old_frame_res.size(), old_frame_res.type());

    while(true)
    {
        Mat frame_src, frame_gray_src;
        Mat frame_res, frame_gray_res;

        if (frame.empty())
            break;

        frame_src = frame;

        cvtColor(frame_src, frame_gray_src, COLOR_BGR2GRAY);
        warpPerspective(frame, frame_res, matH, frame.size(), INTER_LINEAR);
        cvtColor(frame_res, frame_gray_res, COLOR_BGR2GRAY);

        vector<uchar> statusSrc;
        vector<float>errSrc;
        TermCriteria criteriaSrc = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
        vector<uchar> statusRes;
        vector<float>errRes;
        TermCriteria criteriaRes = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

        calcOpticalFlowPyrLK(old_gray_src, frame_gray_src, p0Src, p1Src, statusSrc, errSrc, Size(15, 15), 2, criteriaSrc);
        calcOpticalFlowPyrLK(old_gray_res, frame_gray_res, p0Res, p1Res, statusRes, errRes, Size(15, 15), 2, criteriaRes);

        vector<Point2f> good_new_src;
        vector<Point2f> good_new_res;
        for (uint i = 0; i < p0Src.size(); i++)
        {
            if(statusSrc[i] == 1)
            {
                good_new_src.push_back(p1Src[i]);

                line(maskSrc, p1Src[i], p0Src[i], colors[i], 2);
                circle(frame_src, p1Src[i], 5, colors[i], -1);
            }
        }

        for (uint i = 0; i < p0Src.size(); i++)
        {
            if(statusRes[i] == 1)
            {
                good_new_res.push_back(p1Res[i]);

                line(maskRes, p1Res[i], p0Res[i], colors[i], 2);
                circle(frame_res, p1Res[i], 5, colors[i], -1);
            }
        }

        Mat img_src, img_res;
        add(frame_src, maskSrc, img_src);
        add(frame_res, maskRes, img_src);

        resize(img_src, img_src, Size(resizeWidth, resizeHeight), ratio, ratio, INTER_LINEAR);
        resize(img_res, img_res, Size(resizeWidth, resizeHeight), ratio, ratio, INTER_LINEAR);

        imshow("Source", img_src);
        imshow("Result", img_res);

        int key = waitKey(30);
        if (key >= 0)
        {
            break;
        }

        old_gray_src = frame_gray_src.clone();
        old_gray_res= frame_gray_res.clone();
        p0Src = good_new_src;
        p0Res = good_new_res;
    }
}