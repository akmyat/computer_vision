import sys
import cv2 as cv
import numpy as np

point = (-1, -1)
pts = []
var = 0
drag = 0
mat_final = np.array([])
mat_result = np.array([])

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

color = np.random.randint(0, 255, (100, 3))

VIDEO_FILE = "../../../Data/robot.mp4"
ROTATE = False
CALIBRATION_FILE = "calibration.yml"
CALIBRATE = False


def draw_circle_and_line(x, y):
    global point, pts, mat_result
    mat_result = mat_final.copy()
    point = (x, y)

    if var >= 1:
        cv.line(mat_result, pts[var - 1], point, (0, 255, 0, 255), 2)
    cv.circle(mat_result, point, 2, (0, 255, 0), -1, 8, 0)
    cv.imshow("Source", mat_result)


def mouse_handler(event, x, y, flags, param):
    global point, pts, var, drag, mat_final, mat_result

    if var >= 4:
        return

    if event == cv.EVENT_LBUTTONDOWN:
        drag = 1
        draw_circle_and_line(x, y)

    if event == cv.EVENT_LBUTTONUP and drag:
        drag = 0
        pts.append(point)
        var += 1
        mat_final = mat_result.copy()

        if var >= 4:
            cv.line(mat_final, pts[0], pts[3], (0, 255, 0, 255), 2)
            cv.fillConvexPoly(mat_final, np.array(pts, dtype=np.int32), (0, 120, 0, 20))
        cv.imshow("Source", mat_final)

    if drag:
        draw_circle_and_line(x, y)


if __name__ == "__main__":
    VIDEO_FILE = input("Video file: ")
    CALIBRATION_FILE = input("Calibration file: ")

    cv_file = cv.FileStorage(CALIBRATION_FILE, cv.FILE_STORAGE_READ)
    homography_matrix = cv_file.getNode("H").mat()
    cv_file.release()

    # Open video file
    video_capture = cv.VideoCapture(VIDEO_FILE)
    if not video_capture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        sys.exit('Unable to open input video file')

    width = video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    ratio = 640.0 / width
    dim = (int(width * ratio), int(height * ratio))

    _, frame = video_capture.read()
    old_frame_source = frame
    old_frame_result = cv.warpPerspective(frame, homography_matrix, (frame.shape[1], frame.shape[0]), cv.INTER_LINEAR)

    old_gray_source = cv.cvtColor(old_frame_source, cv.COLOR_BGR2GRAY)
    old_gray_result = cv.cvtColor(old_frame_result, cv.COLOR_BGR2GRAY)

    p0_source = cv.goodFeaturesToTrack(old_gray_source, mask=None, **feature_params)
    p0_result = cv.goodFeaturesToTrack(old_gray_result, mask=None, **feature_params)

    mask_source = np.zeros_like(old_frame_source)
    mask_result = np.zeros_like(old_gray_result)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No frames grabbed!")
            break

        frame_source = frame
        frame_result = cv.warpPerspective(frame, homography_matrix, (frame.shape[1], frame.shape[0]), cv.INTER_LINEAR)

        frame_gray_source = cv.cvtColor(frame_source, cv.COLOR_BGR2GRAY)
        frame_gray_result = cv.cvtColor(frame_result, cv.COLOR_BGR2GRAY)

        p1_source, st_source, err_source = cv.calcOpticalFlowPyrLK(old_gray_source, frame_gray_source, p0_source, None,
                                                                   **lk_params)
        p1_result, st_result, err_result = cv.calcOpticalFlowPyrLK(old_gray_result, frame_gray_result, p0_result, None,
                                                                   **lk_params)

        if p1_source is not None and p1_result is not None:
            good_new_source = p1_source[st_source == 1]
            good_old_source = p0_source[st_source == 1]
            good_new_result = p1_result[st_result == 1]
            good_old_result = p1_result[st_result == 1]

        for i, (new_source, old_source, new_result, old_result) in enumerate(zip(good_new_source, good_old_source,
                                                                                 good_new_result, good_old_result)):
            a_source, b_source = new_source.ravel()
            c_source, d_source = old_source.ravel()
            a_result, b_result = new_result.ravel()
            c_result, d_result = old_result.ravel()

            mask_source = cv.line(mask_source, (int(a_source), int(b_source)), (int(c_source), int(d_source)),
                                  color[i].tolist(), 2)
            mask_result = cv.line(mask_source, (int(a_result), int(b_result)), (int(c_result), int(d_result)),
                                  color[i].tolist(), 2)
            frame_source = cv.circle(frame_source, (int(a_source), int(b_source)), 5, color[i].tolist(), -1)
            frame_result = cv.circle(frame_result, (int(a_result), int(b_result)), 5, color[i].tolist(), -1)

        img_source = cv.add(frame_source, mask_source)
        img_result = cv.add(frame_result, mask_result)

        img_source_resize = cv.resize(img_source, dim)
        img_result_resize = cv.resize(img_result, dim)

        cv.imshow('Source', img_source_resize)
        cv.imshow('Result', img_result_resize)

        key = cv.waitKey(30) & 0xff
        if key == 27:
            break

        old_gray_source = frame_gray_source.copy()
        p0_source = good_new_source.reshape(-1, 1, 2)
        old_gray_result = frame_gray_result.copy()
        p0_result = good_new_result.reshape(-1, 1, 2)

    cv.destroyAllWindows()
