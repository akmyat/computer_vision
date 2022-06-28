#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

// Define the dimensions of checkerboard
int CHECKERBOARD[2] = {8, 11}; // width, height

int main() {
    // Create vector to store vectors of 3D points for each checkerboard image
    vector<vector<Point3f>> objPoints;

    // Create vector to store vectors of 2D points for each checkerboard image
    vector<vector<Point2f>> imgPoints;

    // Define the world coordinates for 3D points
    vector<Point3f> objPoint;
    for (int i=0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
        {
            objPoint.emplace_back(j, i, 0);

        }
    }

    // Extract path of individual image stored in a given directory
    vector<String> images;

    // Path of the folder containing checkerboard images
    string path = "../../../../Data/Lab03/images/*.jpg";

    glob(path, images);

    Mat frame, gray;

    // Vector to store the pixel coordinates of detected checkerboard corners
    vector<Point2f> corner_pts;
    bool success;

    // Looping over all the images in the directory
    for (auto & image : images)
    {
        frame = imread(image);
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Finding checkerboard corners
        // If desired number of corners are found in the image, success will be true
        success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

        /*
         * If desired number of corner are detected, refine the pixel coordinates and
         * display them on the images of checkerboard
         */
        if (success)
        {
            TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);

            // Refine pixel coordinates for given 2D points
            cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);

            // Display the detected corner points on the checkerboard
            drawChessboardCorners(frame, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, true);

            objPoints.push_back(objPoint);
            imgPoints.push_back(corner_pts);
        }

        imshow("Image", frame);
        waitKey(0);
    }
    destroyAllWindows();

    Mat cameraMatrix, distCoefficients, R, T;

    /*
     * Perform camera calibration by passing the value of known 3D points (objPoints)
     * and corresponding pixel coordinates of the detected corners (imgPoints)
     */
    calibrateCamera(objPoints, imgPoints, Size(gray.rows, gray.cols),
                    cameraMatrix, distCoefficients, R, T);
    cout << "Camera Matrix: " << cameraMatrix << endl << endl;
    cout << "Distortion Coefficient: " << distCoefficients << endl << endl;
    cout << "Rotation Vector: " << R << endl << endl;
    cout << "Translation Vector: " << T << endl << endl;

    return 0;
}
