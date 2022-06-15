import cv2
import numpy as np

print(cv2.__version__)
image = np.zeros((300, 600, 3), dtype=np.uint8)

cv2.circle(image, (250, 150), 100, (0, 255, 128), -100)
cv2.circle(image, (350, 150), 100, (255, 255, 255), -100)
cv2.imshow("Display", image)
cv2.waitKey(0)
