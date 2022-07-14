import sys
import time
import cv2 as cv
import numpy as np
from stats import Stats
from utils import draw_bounding_box, draw_statistics, print_statistics, points

akaze_thresh: float = 3e-4      # AKAZE detection threshold set to locate 1000 key points
ransac_thresh: float = 2.5      # RANSAC inlier threshold
nn_match_ratio: float = 0.8     # Nearest-neighbour matching ratio
bb_min_inliers: int = 100       # Minimal number of inliers to draw bounding box
stats_update_period: int = 10   # On-screen statistics are updated every 10 frames


class Tracker:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher
        self.first_frame = None
        self.first_kp = None
        self.first_desc = None
        self.object_bb = None

    def set_first_frame(self, frame, bb, title: str):
        i_size = len(bb)
        stat = Stats()
        pt_contain = np.zeros((i_size, 2))
        i = 0
        for b in bb:
            pt_contain[i, 0] = b[0]
            pt_contain[i, 1] = b[1]
            i += 1

        self.first_frame = frame.copy()
        mat_mask = np.zeros(frame.shape, dtype=np.uint8)
        cv.fillPoly(mat_mask, np.int32([pt_contain]), (255, 0, 0))

        kp = self.detector.detect(self.first_frame, None)
        self.first_kp, self.first_desc = self.detector.compute(self.first_frame, kp)

        res = cv.drawKeypoints(self.first_frame, self.first_kp, None, color=(255, 0, 0),
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        stat.keypoints = len(self.first_kp)
        draw_bounding_box(self.first_frame, bb)

        cv.imshow("key points of {0}".format(title), res)
        cv.waitKey(0)
        cv.destroyWindow("key points of {0}".format(title))

        cv.putText(self.first_frame, title, (0, 60), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 4)
        self.object_bb = bb
        return stat

    def process(self, frame):
        stat = Stats()
        start_time = time.time()
        kp, desc = self.detector.detectAndCompute(frame, None)
        stat.keypoints = len(kp)
        matches = self.matcher.knnMatch(self.first_desc, desc, k=2)

        matched1 = []
        matched2 = []
        matched1_keypoints = []
        matched2_keypoints = []
        good = []

        for i, (m, n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:
                good.append(m)
                matched1_keypoints.append(self.first_kp[matches[i][0].queryIdx])
                matched2_keypoints.append(kp[matches[i][0].trainIdx])
        matched1 = np.float32([self.first_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        matched2 = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        stat.matches = len(matched1)
        homography = None
        if len(matched1) >= 4:
            homography, inlier_mask = cv.findHomography(matched1, matched2, cv.RANSAC, ransac_thresh)
        dt = time.time() - start_time
        stat.fps = 1. / dt
        if len(matched1) < 4 or homography is None:
            res = cv.hconcat([self.first_frame, frame])
            stat.inliers = 0
            stat.ratio = 0
            return res, stat
        inliers1 = []
        inliers2 = []
        inliers1_keypoints = []
        inliers2_keypoints = []
        for i in range(len(good)):
            if inlier_mask[i] > 0:
                new_i = len(inliers1)
                inliers1.append(matched1[i])
                inliers2.append(matched2[i])
                inliers1_keypoints.append(matched1_keypoints[i])
                inliers2_keypoints.append(matched2_keypoints[i])
        inlier_matches = [cv.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0)
                          for idx in range(len(inliers1))]
        inliers1 = np.array(inliers1, dtype=np.float32)
        inliers2 = np.array(inliers2, dtype=np.float32)

        stat.inliers = len(inliers1)
        stat.ratio = stat.inliers * 1.0 / stat.matches
        bb = np.array([self.object_bb], dtype=np.float32)
        new_bb = cv.perspectiveTransform(bb, homography)
        frame_with_bb = frame.copy()
        if stat.inliers >= bb_min_inliers:
            draw_bounding_box(frame_with_bb, new_bb[0])

        res = cv.drawMatches(self.first_frame, inliers1_keypoints, frame_with_bb, inliers2_keypoints, inlier_matches,
                             None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))
        return res, stat

    def get_detector(self):
        return self.detector


if __name__ == "__main__":
    video_name = "../../../Data/robot.mp4"
    video_in = cv.VideoCapture()
    video_in.open(video_name)
    if not video_in.isOpened():
        print("Couldn't open ", video_name)
        sys.exit()

    akaze_stats = Stats()
    orb_stats = Stats()

    akaze = cv.AKAZE_create()
    akaze.setThreshold(akaze_thresh)

    orb = cv.ORB_create()

    matcher = cv.DescriptorMatcher_create("BruteForce-Hamming")

    akaze_tracker = Tracker(akaze, matcher)
    orb_tracker = Tracker(orb, matcher)

    cv.namedWindow(video_name, cv.WINDOW_NORMAL)
    print("\nPress any key to stop the video and select a bounding box")

    key = -1
    while key < 1:
        _, frame = video_in.read()
        if frame is None:
            break
        w, h, ch = frame.shape
        cv.resizeWindow(video_name, w, h)
        cv.imshow(video_name, frame)
        key = cv.waitKey(1)

        print("Select a ROI and then press SPACE or ENTER button!")
        print("Cancel the selection process by pressing c button!")
        uBox = cv.selectROI(video_name, frame);
        bb = []
        bb.append((uBox[0], uBox[1]))
        bb.append((uBox[0] + uBox[2], uBox[0]))
        bb.append((uBox[0] + uBox[2], uBox[0] + uBox[3]))
        bb.append((uBox[0], uBox[0] + uBox[3]))

        stat_a = akaze_tracker.set_first_frame(frame, bb, "AKAZE")
        stat_o = orb_tracker.set_first_frame(frame, bb, "ORB")

        akaze_draw_stats = stat_a.copy()
        orb_draw_stats = stat_o.copy()

        i = 0
        video_in.set(cv.CAP_PROP_POS_FRAMES, 0)
        while True:
            i += 1
            update_stats = (i % stats_update_period == 0)
            _, frame = video_in.read()
            if frame is None:
                # End of video
                break
            akaze_res, stat = akaze_tracker.process(frame)
            akaze_stats + stat
            if (update_stats):
                akaze_draw_stats = stat
            orb.setMaxFeatures(stat.keypoints)
            orb_res, stat = orb_tracker.process(frame)
            orb_stats + stat
            if (update_stats):
                orb_draw_stats = stat
            draw_statistics(akaze_res, akaze_draw_stats)
            draw_statistics(orb_res, orb_draw_stats)
            res_frame = cv.vconcat([akaze_res, orb_res])
            # cv2.imshow(video_name, akaze_res)
            cv.imshow(video_name, res_frame)
            if (cv.waitKey(1) == 27):  # quit on ESC button
                break

        akaze_stats / (i - 1)
        orb_stats / (i - 1)
        print_statistics("AKAZE", akaze_stats)
        print_statistics("ORB", orb_stats)