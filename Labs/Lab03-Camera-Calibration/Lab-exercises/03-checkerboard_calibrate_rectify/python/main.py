import sys
import cv2 as cv
# from capture_video import capture_video, convert_video_to_images
from calibrate_camera import CameraCalibration
from homography import Homography

PATH = "../../../../Data/Lab03/checkerboard/"
CALIBRATION_FILE = "checkerboard.mp4"
VIDEO_FILE = "car.mp4"

# Capture the checkerboard video
# capture_video(PATH + CALIBRATION_FILE)

# Convert the checkerboard video to images
# convert_video_to_images(CALIBRATION_FILE, PATH)

# Calibrate the camera
camera_calibration = CameraCalibration()
camera_calibration.calibrate(PATH + "images/")
camera_calibration.write_calibration_file("camera_calibration.yml")

print("Camera matrix: ", camera_calibration.camera_mtx)
print("Distortion coefficient: ", camera_calibration.distortion_coefficient)
print("Rotation vectors: ", camera_calibration.rotation_vectors)
print("Translation vectors: ", camera_calibration.translation_vectors)

# Calculate homography
homography_data = Homography()
homography_data.calculate(PATH + VIDEO_FILE)
homography_data.write("homography.yml")

print("Estimated Homography matrix: \n", homography_data.mat_h)

# Show undistorted and rectify images
video_capture = cv.VideoCapture(PATH + VIDEO_FILE)
if not video_capture.isOpened():
    print("ERROR! Unable to open video file ", VIDEO_FILE)
    sys.exit()
width = video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
ratio = 640.0 / width
dim = (int(width * ratio), int(height * ratio))

duration = 0
while True:
    _, frame = video_capture.read()
    if frame is None:
        break

    # Undistorted and Cropped the frame
    undistorted_frame = cv.undistort(frame, camera_calibration.camera_mtx,
                                     camera_calibration.distortion_coefficient)
    h, w = frame.shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_calibration.camera_mtx,
                                                       camera_calibration.distortion_coefficient,
                                                       (w, h), 1, (w, h))
    x, y, w, h = roi
    cropped_undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    # Show the original and undistorted frames
    cv.namedWindow("Original Frame", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.imshow("Original Frame", frame)
    cv.namedWindow("Undistorted Frame", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.imshow("Undistorted Frame", cropped_undistorted_frame)

    # Show the rectified original and undistorted frames
    result_original = cv.warpPerspective(frame, homography_data.mat_h,
                                         (int(homography_data.width_out), int(homography_data.height_out)),
                                         cv.INTER_LINEAR)
    result_undistorted = cv.warpPerspective(cropped_undistorted_frame, homography_data.mat_h,
                                            (int(homography_data.width_out), int(homography_data.height_out)),
                                            cv.INTER_LINEAR)
    cv.namedWindow("Rectify Original", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.imshow("Rectify Original", result_original)
    cv.namedWindow("Rectify Undistorted", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.imshow("Rectify Undistorted", result_undistorted)

    cv.waitKey(duration)
    if duration == 0:
        duration = 30
cv.destroyAllWindows()
