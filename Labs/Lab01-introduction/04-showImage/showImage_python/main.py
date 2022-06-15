import cv2 as cv

IMAGE_FILE = "/mnt/ntfs/Data/code/CV/CV/Labs/02-Homographies/img/sample.jpg"

srcImage = cv.imread(IMAGE_FILE)
if not srcImage.data:
    print("ERROR!")

cv.imshow("srcImage", srcImage)
cv.waitKey(0)

greyMat = cv.cvtColor(srcImage, cv.COLOR_BGR2GRAY)
cv.imshow("greyImage", greyMat)
cv.waitKey(0)
