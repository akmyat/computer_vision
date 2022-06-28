from calibrate_camera import CameraCalibration

camera_calibration = CameraCalibration()

camera_calibration.calibrate("../../../../Data/Lab03/images/")
print("Camera matrix: ", camera_calibration.camera_mtx)
print("Distortion coefficient: ", camera_calibration.distortion_coefficient)
print("Rotation vectors: ", camera_calibration.rotation_vectors)
print("Translation vectors: ", camera_calibration.translation_vectors)

camera_calibration.write_calibration_file("checkerboard_calibration_file.yml")
print()

camera_calibration.read_calibration_file("checkerboard_calibration_file.yml")
print("Camera matrix: ", camera_calibration.camera_mtx)
print("Distortion coefficient: ", camera_calibration.distortion_coefficient)
print("Rotation vectors: ", camera_calibration.rotation_vectors)
print("Translation vectors: ", camera_calibration.translation_vectors)
