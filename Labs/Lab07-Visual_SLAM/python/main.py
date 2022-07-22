from calibrate_camera import CameraCalibration

# ----------- Calibrate Camera -----------
CALIBRATE_PATH = "../../../Data/Lab03/checkerboard/images/"
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