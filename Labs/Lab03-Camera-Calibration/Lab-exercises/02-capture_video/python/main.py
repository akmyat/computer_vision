import cv2 as cv

OUTPUT_FILE_NAME = "checkerboard.mp4"

video_capture = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*"mp4v")
out = cv.VideoWriter(OUTPUT_FILE_NAME, fourcc, 20.0, (640, 480))

while True:
    ret, frame = video_capture.read()

    out.write(frame)

    cv.imshow('Web Camera', frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
video_capture.release()
out.release()
