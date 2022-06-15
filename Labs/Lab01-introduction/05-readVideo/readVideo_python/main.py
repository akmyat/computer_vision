import cv2 as cv
import sys

VIDEO_FILE = "../../../data/robot.mp4"

if __name__ == "__main__":
    key = -1

    # Open input video file
    videoCapture = cv.VideoCapture(VIDEO_FILE)
    if not videoCapture.isOpened():
        print("Error. Unable to open input video file.")
        sys.exit()

    height = videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)

    while key < 0:
        # Get next frame
        _, matFrameCapture = videoCapture.read()

        if matFrameCapture is None:
            break   # End of video file

        # Resize the frame
        ratio = 480.0 / height
        dim = (int(width * ratio), int(height * ratio))
        matFrameCapture = cv.resize(matFrameCapture, dim)

        # Show the frame
        cv.imshow(VIDEO_FILE, matFrameCapture)

        # Get key input
        key = cv.waitKey(30)

