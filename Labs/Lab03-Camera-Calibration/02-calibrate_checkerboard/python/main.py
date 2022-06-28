import glob
import cv2 as cv
import numpy as np


# Define the dimensions of checkerboard
CHECKERBOARD = (8, 11)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Create vector to store vectors of 3D points for each checkerboard image
obj_points = []
# Create vector to store vectors of 2D points for each checkerboard image
img_points = []

# Define the world coordinates for 3D points
obj_point = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float32)
obj_point[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob("../../../Data/Lab03/checkerboard/images/*.jpg")
img = None
gray = None
for f_name in images:

    img = cv.imread(f_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK
                                            + cv.CALIB_CB_NORMALIZE_IMAGE)
    """
        If desired number of corner are detected, refine the pixel coordinates and display them 
        on the images of checkerboard
    """
    if ret:
        obj_points.append(obj_point)

        # Refining pixel coordinates for given 2D points.
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        img_points.append(corners2)

        # Draw and display the corners
        img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(0)

cv.destroyAllWindows()

h, w = img.shape[:2]
"""
    Performing camera calibration by passing the value of known 3D points (obj_points) and 
    corresponding pixel coordinates of the detected corners (img_points)
"""
ret, mtx, dist, r_vectors, t_vectors = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("Camera matrix: \n")
print(mtx)
print("Distortion Coefficient: \n")
print(dist)
print("Rotation vectors: \n")
print(r_vectors)
print("Translation vectors: \n")
print(t_vectors)

# Show images of undistorted
for f_name in images:
    img = cv.imread(f_name)
    res = cv.undistort(img, mtx, dist)
    cv.imshow('img', img)
    cv.imshow('res', res)
    cv.waitKey(0)

# Use ROI obtained above to crop the result
print("Undistorted using ROI")
for f_name in images:
    print(f_name)

    img = cv.imread(f_name)

    h, w = img.shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    res = cv.undistort(img, mtx, dist)

    # crop the image
    x, y, w, h = roi
    res = res[y:y+h, x:x+w]

    cv.imshow('img', img)
    cv.imshow('res', res)
    cv.waitKey(0)


# Find a mapping function from the distorted image to undistorted image.
# Then use the remap function.
print("Find mapping from distorted to undistorted image.\nThen use the remap function.")
for f_name in images:
    print(f_name)

    img = cv.imread(f_name)
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    map_x, map_y = cv.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
    res = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

    # Crop the image
    x, y, w, h = roi
    res = res[y:y+h, x:x+w]

    cv.imshow('img', img)
    cv.imshow('res', res)
    cv.waitKey(0)

mean_error = 0
for i in range(len(obj_points)):
    img_points2, _ = cv.projectPoints(obj_points[i], r_vectors[i], t_vectors[i], mtx, dist)
    error = cv.norm(img_points[i], img_points2, cv.NORM_L2) / len(img_points2)
    mean_error += error
print(f"Total error: {mean_error / len(obj_points)}")
