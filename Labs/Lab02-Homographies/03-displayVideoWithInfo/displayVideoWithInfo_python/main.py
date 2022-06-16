import sys
import cv2 as cv

VIDEO_FILE = "../../../Data/robot.mp4"
ROTATE = False

if __name__ == "__main__":
    key = ord(" ")

    # Open video file
    videoCapture = cv.VideoCapture(VIDEO_FILE)
    height = videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)
    totalFrameNum = int(videoCapture.get(cv.CAP_PROP_FRAME_COUNT))
    if not videoCapture.isOpened():
        print("ERROR! Unable to open video file {}".format(VIDEO_FILE))
        sys.exit()

    # Capture loop
    while True:
        # Get the next frame when press <space>
        if key == ord(" "):
            _, matFrameCapture = videoCapture.read()
            currentFrameNum = int(videoCapture.get(cv.CAP_PROP_POS_FRAMES))
            if matFrameCapture is None:
                # End of video
                break

            # Rotate Video
            if ROTATE:
                # Rotate 180 degree and put image to matFrameDisplay
                matFrameDisplay = cv.rotate(matFrameCapture, cv.ROTATE_180)
            else:
                matFrameDisplay = matFrameCapture

            # Resize the frame
            height_ratio = 768.0 / height
            width_ratio = 1366.0 / width
            down_height = int(height * height_ratio)
            down_width = int(width * width_ratio)
            matFrameDisplay = cv.resize(matFrameDisplay,
                                        (down_width, down_height),
                                        width_ratio, height_ratio,
                                        cv.INTER_LINEAR)

            # 1366p x 768p frame display in resizeable, keep aspect ratio, and show expended GUI
            cv.namedWindow("ROBOT.MP4", flags=cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)

            # Display the frame
            cv.imshow("ROBOT.MP4", matFrameDisplay)

            # Display overlay explanatory information
            explanatory_info = f"{currentFrameNum} / {totalFrameNum} " \
                               f"frames. Press <space> for next frame. Press <q> to quit."
            cv.displayOverlay("ROBOT.MP4", explanatory_info)

        # Quit when press <q>
        if key == ord('q'):
            break

        # Exit program on window close
        if cv.getWindowProperty("ROBOT.MP4", cv.WND_PROP_VISIBLE) != 1:
            break

        key = cv.pollKey()
    cv.destroyAllWindows()
