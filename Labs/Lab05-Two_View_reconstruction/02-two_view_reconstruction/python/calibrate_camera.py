import glob
import cv2 as cv
import numpy as np


class CameraCalibration:
    camera_mtx = None
    distortion_coefficient = None
    rotation_vectors = None
    translation_vectors = None

    # Define the dimensions of checkerboard
    __CHECKERBOARD = (6, 9)
    __criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Create vector to store vectors of 3D points for each checkerboard image
    __obj_points = []

    # Create vector to store vectors of 2D points for each checkerboard image
    __img_points = []

    # Define the world coordinates for 3D points
    __obj_point = np.zeros((1, __CHECKERBOARD[0] * __CHECKERBOARD[1], 3), dtype=np.float32)
    __obj_point[0, :, :2] = np.mgrid[0:__CHECKERBOARD[0], 0:__CHECKERBOARD[1]].T.reshape(-1, 2)
    __prev_img_shape = None

    __img = None
    __gray = None

    def __init__(self, file_name=None):
        if file_name is not None:
            self.read_calibration_file(file_name)

    def calibrate(self, file_path):
        images = glob.glob(file_path + "*.jpg")

        for f_name in images:
            self.__img = cv.imread(f_name)
            self.__gray = cv.cvtColor(self.__img, cv.COLOR_BGR2GRAY)

            # Find the checkerboard corners
            ret, corners = cv.findChessboardCorners(self.__gray, self.__CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH
                                                    + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

            # If the desired number of corner are detected
            if ret:
                self.__obj_points.append(self.__obj_point)

                # Refining pixel coordinates for given 2D points.
                corners2 = cv.cornerSubPix(self.__gray, corners, (9, 9), (-1, -1), self.__criteria)

                self.__img_points.append(corners2)

                # Draw and display the corners
                self.__img = cv.drawChessboardCorners(self.__img, self.__CHECKERBOARD, corners2, True)

            cv.imshow('img', self.__img)
            cv.waitKey(30)
        cv.destroyAllWindows()

        # Calibrate the camera
        ret, self.camera_mtx, self.distortion_coefficient, self.rotation_vectors, self.translation_vectors = \
            cv.calibrateCamera(self.__obj_points, self.__img_points, self.__gray.shape[::-1], None, None)

    def write_calibration_file(self, file_name):
        file_storage = cv.FileStorage(file_name, cv.FILE_STORAGE_WRITE)

        if not file_storage.isOpened():
            return False

        file_storage.write("MTX", self.camera_mtx)
        file_storage.write("DIST", self.distortion_coefficient)
        for i in range(10):
            file_storage.write("R" + str(i), self.rotation_vectors[i])
            file_storage.write("T" + str(i), self.translation_vectors[i])
        file_storage.release()

        return True

    def read_calibration_file(self, file_name):
        file_storage = cv.FileStorage(file_name, cv.FILE_STORAGE_READ)
        if not file_storage.isOpened():
            return False

        self.camera_mtx = file_storage.getNode("MTX").mat()
        self.distortion_coefficient = file_storage.getNode("DIST").mat()
        r_vectors = tuple()
        t_vectors = tuple()
        for i in range(10):
            r_vectors += (file_storage.getNode("R" + str(i)).mat(),)
            t_vectors += (file_storage.getNode("T" + str(i)).mat(),)
        self.rotation_vectors = r_vectors
        self.translation_vectors = t_vectors
        file_storage.release()

        return True
