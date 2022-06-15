import sys
import cv2 as cv

VIDEO_FILE = "../../../Data/robot.mp4"
ROTATE = False

if __name__ == "__main__":
    key = -1

    # Open video file
    videoCapture = cv.VideoCapture(VIDEO_FILE)
    height = videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)

    if not videoCapture.isOpened():
        print("ERROR! Unable to open video file {}".format(VIDEO_FILE))
        sys.exit()

    # Capture loop
    while key != ord(' '):
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:
            # End of video
            break

        # Rotate Video
        if ROTATE:
            # Rotate 180 degree and put image to matFrameDisplay
            _, matFrameDisplay = cv.rotate(matFrameCapture, cv.ROTATE_180)
        else:
            matFrameDisplay = matFrameCapture

        # Resize the frame
        ratio = 480.0 / height
        down_height = int(height * ratio)
        down_width = int(width * ratio)
        matFrameDisplay = cv.resize(matFrameDisplay,
                                    (down_width, down_height),
                                    ratio, ratio, cv.INTER_LINEAR)

        # Display the frame
        cv.imshow(VIDEO_FILE, matFrameDisplay)

        key = cv.waitKey(30)
