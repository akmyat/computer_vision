import cv2 as cv
import numpy as np
from calibrate_camera import CameraCalibration

left_img = cv.imread("../../../Data/Lab05/test/image0.jpg")
right_img = cv.imread("../../../Data/Lab05/test/image1.jpg")

akaze_thresh: float = 3e-4
ransac_thresh: float = 2.5
nn_match_ratio: float = 0.8
bb_min_inliers: int = 100
stats_update_period: int = 10

# Calibrate camera
CALIBRATE_PATH = "../../../Data/Lab03/checkerboard/images/"
camera_calibration = CameraCalibration()
# camera_calibration.calibrate(CALIBRATE_PATH)
# camera_calibration.write_calibration_file("camera_calibration.yml")
camera_calibration.read_calibration_file("camera_calibration.yml")
camera_mtx = camera_calibration.camera_mtx
camera_dist = camera_calibration.distortion_coefficient


# Get Bounding Box
def get_ubox(image):
    u_box = cv.selectROI("Image", image)
    bb = list()
    bb.append((u_box[0], u_box[1]))
    bb.append((u_box[0]+u_box[2], u_box[0]))
    bb.append((u_box[0]+u_box[2], u_box[0]+u_box[3]))
    bb.append((u_box[0], u_box[0]+u_box[3]))
    return bb


left_bb = get_ubox(left_img)
right_bb = get_ubox(right_img)


# Detect Key Points
def detect_and_draw_key_points(detector, img, bounding_box):
    i_size = len(bounding_box)
    pt_contain = np.zeros((i_size, 2))
    for index, b in enumerate(bounding_box):
        pt_contain[index, 0] = b[0]
        pt_contain[index, 1] = b[1]

    frame = img.copy()
    mat_mask = np.zeros(img.shape, dtype=np.uint8)
    cv.fillPoly(mat_mask, np.int32([pt_contain]), (255, 0, 0))

    kp = detector.detect(frame, None)
    kps, desc = detector.compute(frame, kp)

    res = cv.drawKeypoints(img, kps, None, color=(255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return res, kps, desc


detector1 = cv.AKAZE_create()
detector1.setThreshold(akaze_thresh)
left_kps_img, left_detector1_kps, left_detector1_desc = detect_and_draw_key_points(detector1, left_img, left_bb)
right_kps_img, right_detector1_kps, right_detector1_desc = detect_and_draw_key_points(detector1, right_img, right_bb)
detector1_kps_img = cv.hconcat([left_kps_img, right_kps_img])

detector2 = cv.ORB_create()
left_kps_img, left_detector2_kps, left_detector2_desc = detect_and_draw_key_points(detector2, left_img, left_bb)
right_kps_img, right_detector2_kps, right_detector2_desc = detect_and_draw_key_points(detector2, right_img, right_bb)
detector2_kps_img = cv.hconcat([left_kps_img, right_kps_img])

kps_res = cv.vconcat([detector1_kps_img, detector2_kps_img])
cv.namedWindow("Key Points", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
cv.imshow("Key Points", kps_res)
cv.waitKey(0)
cv.destroyWindow("Key Points")


# Match features
def draw_match_features(img0, img1, kps0, kps1, desc0, desc1, camera_matrix=None, camera_dist_coeffs=None):
    matcher = cv.DescriptorMatcher_create("BruteForce-Hamming")
    matches = matcher.knnMatch(desc0, desc1, k=2)
    good_matches = list()
    if camera_matrix is not None:
        matched1_kps = list()
        matched2_kps = list()
        for i, (m, n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:
                good_matches.append(m)
                matched1_kps.append(kps0[m.queryIdx])
                matched2_kps.append(kps1[m.trainIdx])
        matched1 = np.array([kps0[m.queryIdx].pt for m in good_matches])
        matched2 = np.array([kps1[m.trainIdx].pt for m in good_matches])

        # Normalized Points
        norm_matched1 = cv.undistortPoints(matched1, camera_matrix, camera_dist_coeffs)
        norm_matched2 = cv.undistortPoints(matched2, camera_matrix, camera_dist_coeffs)
        K = camera_matrix
        E, mask = cv.findEssentialMat(norm_matched1, norm_matched2, K, method=cv.RANSAC, prob=0.999,
                                      threshold=ransac_thresh)

        inliers1 = list()
        inliers2 = list()
        inlier_matches = list()
        for i in range(len(matched1)):
            if mask[i]:
                new_i = len(inliers1)
                inliers1.append(matched1_kps[i])
                inliers2.append(matched2_kps[i])
                inlier_matches.append(cv.DMatch(new_i, new_i, 0))

    else:
        for i, (m, n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:
                good_matches.append(m)
        inliers1 = kps0
        inliers2 = kps1
        inlier_matches = good_matches

    matched_res = cv.drawMatches(img0, inliers1, img1, inliers2, inlier_matches, None, matchColor=(255, 0, 0),
                                 singlePointColor=(255, 0, 0))
    if camera_matrix is not None:
        return matched_res, E, inliers1, inliers2, inlier_matches
    else:
        return matched_res, inlier_matches


# Feature Point matching using AKAZE and ORB
detector1_res, detector1_matches = draw_match_features(left_img, right_img, left_detector1_kps, right_detector1_kps,
                                                       left_detector1_desc, right_detector1_desc)
detector2_res, detector2_matches = draw_match_features(left_img, right_img, left_detector2_kps, right_detector2_kps,
                                                       left_detector2_desc, right_detector2_desc)

match_feature_res = cv.vconcat([detector1_res, detector2_res])
cv.namedWindow("Result", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
cv.imshow("Result", match_feature_res)
cv.waitKey(0)

# Feature Point matching improve with essential matrix
detector1_res_with_E, detector1_E, detector1_inliers1, detector1_inliers2, _ = draw_match_features(left_img, right_img, left_detector1_kps,
                                                                 right_detector1_kps, left_detector1_desc,
                                                                 right_detector1_desc, camera_mtx, camera_dist)
detector2_res_with_E, detector2_E, detector2_inliers1, detector2_inliers2, _ = draw_match_features(left_img, right_img, left_detector2_kps,
                                                                 right_detector2_kps, left_detector2_desc,
                                                                 right_detector2_desc, camera_mtx, camera_dist)
match_feature_res_with_E = cv.vconcat([detector1_res_with_E, detector2_res_with_E])
cv.namedWindow("Result", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
cv.imshow("Result", match_feature_res_with_E)
cv.waitKey(0)


# Verify X^T K^-T E K^-1 X' = 0
def verify_essential_matrix(points1, points2, essential_mtx, camera_matrix):
    left_point = np.array(points1[0].pt + (1, )).reshape(3, 1)
    right_point = np.array(points2[0].pt + (1,)).reshape(3, 1)
    K = camera_matrix
    K_inv = np.linalg.inv(K)
    E = essential_mtx

    value = right_point.T @ K_inv.T @ E @ K_inv @ left_point
    if value < 0:
        print("X^T K^-T E K^-1 X' = 0")
        return True
    else:
        print(f"X^T K^-T E K^-1 X' = {value}")
        return False


assert verify_essential_matrix(detector1_inliers1, detector1_inliers2, detector1_E, camera_mtx)
assert verify_essential_matrix(detector2_inliers1, detector2_inliers2, detector2_E, camera_mtx)

# Find Epi-polar lines
pt1 = np.concatenate((np.int32(detector1_inliers1[0].pt), np.ones(1))).reshape(3, 1)
pt2 = np.concatenate((np.int32(detector1_inliers2[0].pt), np.ones(1))).reshape(3, 1)
E = detector1_E
a = E @ pt1
a /= a[2]
print(a[0]*480, a[1]*640, a[2])
print(pt2)
# cv.imshow("Epi-polar lines", img)
# cv.waitKey(0)
# cv.destroyWindow("Epi-polar lines")
