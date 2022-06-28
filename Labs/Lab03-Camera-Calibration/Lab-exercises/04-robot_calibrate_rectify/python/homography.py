import cv2 as cv
import numpy as np


class Homography:
    mat_h = np.zeros((3, 3))
    width_out: int
    height_out:  int
    c_points: int
    a_points: list = []

    __point = (-1, -1)
    __pts = []
    __var = 0
    __drag = 0
    __mat_final = np.array([])
    __mat_result = np.array([])

    def __init__(self, homography_file=None):
        self.c_points = 0
        if homography_file is not None:
            self.read(homography_file)

    def read(self, homography_file: str):
        file_storage = cv.FileStorage(homography_file, cv.FILE_STORAGE_READ)
        if not file_storage.isOpened():
            return False

        self.c_points = 0
        for i in range(4):
            point = file_storage.getNode("aPoints" + str(i))
            self.a_points.append(point.mat())
            self.c_points += 1

        self.mat_h = file_storage.getNode("matH").mat()
        self.width_out = int(file_storage.getNode("widthOut").real())
        self.height_out = int(file_storage.getNode("heightOut").real())
        file_storage.release()
        return True

    def write(self, homography_file):
        file_storage = cv.FileStorage(homography_file, cv.FILE_STORAGE_WRITE)
        if not file_storage.isOpened():
            return False

        for i in range(4):
            file_storage.write("aPoints" + str(i), self.a_points[i])

        file_storage.write("matH", self.mat_h)
        file_storage.write("widthOut", self.width_out)
        file_storage.write("heightOut", self.height_out)
        file_storage.release()

        return True

    def __draw_circle_and_line(self, x, y):
        self.__mat_result = self.__mat_final.copy()
        self.__point = (x, y)

        if self.__var >= 1:
            cv.line(self.__mat_result, self.__pts[self.__var - 1], self.__point, (0, 255, 0, 255), 2)
        cv.circle(self.__mat_result, self.__point, 2, (0, 255, 0), -1, 8, 0)
        cv.imshow("Source", self.__mat_result)

    def __mouse_handler(self, event, x, y, flags, param):
        if self.__var >= 4:
            return

        if event == cv.EVENT_LBUTTONDOWN:
            self.__drag = 1
            self.__draw_circle_and_line(x, y)

        if event == cv.EVENT_LBUTTONUP and self.__drag:
            self.__drag = 0
            self.__pts.append(self.__point)
            self.__var += 1
            self.__mat_final = self.__mat_result.copy()

            if self.__var >= 4:
                cv.line(self.__mat_final, self.__pts[0], self.__pts[3], (0, 255, 0, 255), 2)
                cv.fillConvexPoly(self.__mat_final, np.array(self.__pts, dtype=np.int32), (0, 120, 0, 20))
            cv.imshow("Source", self.__mat_final)

        if self.__drag:
            self.__draw_circle_and_line(x, y)

    def calculate(self, file_name):
        mat_pause_screen = np.array([])
        mat_frame_capture = np.array([])
        key = -1

        # --------------------- [STEP 1: Make video capture from file] ---------------------
        # Open video file
        video_capture = cv.VideoCapture(file_name)
        if not video_capture.isOpened():
            print("ERROR! Unable to open input video file ", file_name)
            return False

        width = video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        ratio = 640.0 / width
        dim = (int(width * ratio), int(height * ratio))

        while key < 0:
            # Get the next frame
            _, mat_frame_capture = video_capture.read()
            if mat_frame_capture is None:
                break

            mat_frame_display = cv.resize(mat_frame_capture, dim)

            cv.imshow("Original", mat_frame_display)
            key = cv.waitKey(30)

        # --------------------- [STEP 2: pause the screen and show an image] ---------------------
            if key >= 0:
                mat_pause_screen = mat_frame_capture
                self.__mat_final = mat_pause_screen.copy()

        cv.destroyAllWindows()

        # --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
        if mat_frame_capture is not None:
            self.__var = 0
            self.__pts.clear()
            cv.namedWindow("Source", cv.WINDOW_GUI_NORMAL)
            cv.setMouseCallback("Source", self.__mouse_handler)
            cv.imshow("Source", mat_pause_screen)
            cv.waitKey(0)
            cv.destroyWindow("Source")

            if len(self.__pts) == 4:
                src = np.array(self.__pts).astype(np.float32)

                reals = np.array([
                    (800, 800),
                    (1000, 800),
                    (1000, 1000),
                    (800, 1000)
                ], dtype=np.float32)

        # --------------------- [STEP 4: Calculate Homography] ---------------------
                homography_matrix = cv.getPerspectiveTransform(src, reals)

        # --------------------- [STEP 4: Calculate Homography] ---------------------
                self.mat_h = homography_matrix
                self.c_points = 0
                for i in range(4):
                    self.a_points.append(src[i])
                    self.c_points += 1
                self.width_out = int(width)
                self.height_out = int(height)

                return True
        else:
            return False
