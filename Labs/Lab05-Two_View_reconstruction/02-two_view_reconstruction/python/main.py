from calibrate_camera import CameraCalibration
from stereo_vision import StereoVision
import cv2 as cv
import pandas as pd
import plotly.express as px


def show_result_img(window_name, image):
    cv.namedWindow(window_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.imshow(window_name, image)
    cv.waitKey(0)
    cv.destroyWindow(window_name)


# ----------- Calibrate Camera -----------
CALIBRATE_PATH = "../../../Data/Lab03/robot/images/"
camera = CameraCalibration()
# camera.calibrate(CALIBRATE_PATH)
# camera.write_calibration_file("camera_calibration.yml")
camera.read_calibration_file("camera_calibration.yml")

camera_mtx = camera.camera_mtx
camera_dist = camera.distortion_coefficient
print("\nCamera Matrix")
print(camera_mtx)
print("\nCamera Distortion")
print(camera_dist)

# ----------- Read two images -----------
left_img = cv.imread("../../../Data/Lab05/robot/image0.png")
right_img = cv.imread("../../../Data/Lab05/robot/image1.png")
img = cv.hconcat([left_img, right_img])
show_result_img("Two Frames", img)

# ----------- Create detector -----------
detector1 = cv.AKAZE_create()
detector2 = cv.ORB_create()

# ----------- Create Stereo Vision Instance -----------
stereo_vision1 = StereoVision(left_img, right_img, camera_mtx, camera_dist, detector1)
stereo_vision2 = StereoVision(left_img, right_img, camera_mtx, camera_dist, detector2)

# ----------- Detect Key Points in image -----------
akaze_kps_img, akaze_img1_kps, akaze_img1_desc, akaze_img2_kps, akaze_img2_desc = stereo_vision1.detect_key_points()
orb_kps_img, orb_img1_kps, orb_img1_desc, orb_img2_kps, orb_img2_desc = stereo_vision2.detect_key_points()
cv.putText(akaze_kps_img, "AKAZE", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
cv.putText(orb_kps_img, "ORB", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
img = cv.vconcat([akaze_kps_img, orb_kps_img])

show_result_img("Detect Key Points", img)

# ----------- Match Key Points -----------
nn_match_ratio: float = 0.8
print("\nAKAZE")
akaze_matched_img, akaze_pts1, akaze_pts2, akaze_good_matches = stereo_vision1.match_key_points(nn_match_ratio)
print("\nORB")
orb_matched_img, orb_pts1, orb_pts2, orb_good_matches = stereo_vision2.match_key_points(nn_match_ratio)
cv.putText(akaze_matched_img, "AKAZE", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
cv.putText(orb_matched_img, "ORB", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
img = cv.vconcat([akaze_matched_img, orb_matched_img])
show_result_img("Match Key Points", img)

# ----------- Find Essential Matrix -----------
akaze_essential_img, akaze_essential_matrix, akaze_essential_pts1, akaze_essential_pts2, akaze_new_good_matches = \
    stereo_vision1.find_essential_matrix(akaze_pts1, akaze_pts2)
orb_essential_img, orb_essential_matrix, orb_essential_pts1, orb_essential_pts2, orb_new_good_matches = \
    stereo_vision2.find_essential_matrix(orb_pts1, orb_pts2)

cv.putText(akaze_essential_img, "AKAZE", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
cv.putText(orb_essential_img, "ORB", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
img = cv.vconcat([akaze_essential_img, orb_essential_img])
show_result_img("Match Key Points Using Essential Matrix", img)

# ----------- Verify Essential Matrix -----------
print("\nVerify essential matrix for AKAZE")
akaze_result = stereo_vision1.verify_essential_matrix(akaze_essential_pts1, akaze_essential_pts2, akaze_essential_matrix)
print(akaze_result)
print("\nVerify essential matrix for ORB")
orb_result = stereo_vision1.verify_essential_matrix(orb_essential_pts1, orb_essential_pts2, orb_essential_matrix)
print(orb_result)

# ----------- Undistorted Matches -----------
akaze_undistorted_matches = stereo_vision1.undistorted_matches(left_img, right_img, akaze_img1_kps, akaze_img2_kps, akaze_new_good_matches)
orb_undistorted_matches = stereo_vision2.undistorted_matches(left_img, right_img, orb_img1_kps, orb_img2_kps, orb_new_good_matches)

cv.putText(akaze_undistorted_matches, "AKAZE", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
cv.putText(orb_undistorted_matches, "ORB", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
img = cv.vconcat([akaze_undistorted_matches, orb_undistorted_matches])
show_result_img("Undistorted Match Key Points", img)

# ----------- Undistorted Epipolar line -----------
akaze_undistorted_epipolar_lines = stereo_vision1.undistorted_epipolar_lines(left_img, right_img, akaze_essential_pts1,
                                                                      akaze_essential_pts2)

orb_undistorted_epipolar_lines = stereo_vision2.undistorted_epipolar_lines(left_img, right_img, orb_essential_pts1,
                                                                             orb_essential_pts2)
cv.putText(akaze_undistorted_epipolar_lines, "AKAZE", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
cv.putText(orb_undistorted_epipolar_lines, "ORB", (0, 100), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
img = cv.vconcat([akaze_undistorted_epipolar_lines, orb_undistorted_epipolar_lines])
show_result_img("Undistorted Epipolar lines", img)

# ----------- Decompose essential matrix -----------
print("\nAKAZE")
akaze_rotation, akaze_transition = stereo_vision1.decompose_essential_matrix()
print("\nORB")
orb_rotation, orb_transition = stereo_vision2.decompose_essential_matrix()

# ----------- Recover Relative Poses -----------
akaze_fundamental_matrix = stereo_vision1.find_fundamental_matrix(akaze_essential_matrix)
akaze_points_3d = stereo_vision1.recover_relative_pose(akaze_essential_pts1,  akaze_essential_pts2, akaze_fundamental_matrix,
                                                 akaze_essential_matrix)

df = pd.DataFrame(akaze_points_3d, columns=["X", "Y", "Z"])
fig = px.scatter_3d(df, x="X", y="Y", z="Z")
fig.show()

orb_fundamental_matrix = stereo_vision2.find_fundamental_matrix(orb_essential_matrix)
orb_points_3d = stereo_vision2.recover_relative_pose(orb_essential_pts1,  orb_essential_pts2, orb_fundamental_matrix,
                                                       orb_essential_matrix)

df = pd.DataFrame(orb_points_3d, columns=["X", "Y", "Z"])
fig = px.scatter_3d(df, x="X", y="Y", z="Z")
fig.show()
