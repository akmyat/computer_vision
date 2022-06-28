import cv2 as cv

video_capture = cv.VideoCapture()
video_capture.open(0, cv.CAP_ANY)
if not video_capture.isOpened():
    print("ERROR! Unable to open camera\n")
    exit()

print("Start grabbing")
print("Press s to save images and q to terminate")

frame_add = 0
while True:
    _, frame = video_capture.read()
    if frame is None:
        print("ERROR! Blank frame grabbed\n")
        exit()

    cv.imshow("Live", frame)

    iKey = cv.waitKey(5)
    if iKey == ord('s') or iKey == ord('S'):
        cv.imwrite("../../../Data/Lab03/images/frame" + str(frame_add) + ".jpg", frame)
        frame_add += 1
        print("Frame: ", frame_add, " has been saved.")
    elif iKey == ord('q') or iKey == ord('Q'):
        break
