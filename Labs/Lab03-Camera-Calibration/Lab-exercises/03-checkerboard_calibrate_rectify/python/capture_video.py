import cv2 as cv


def capture_video(file_name: str):
    video_capture = cv.VideoCapture(0)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(file_name, fourcc, 20.0, (640, 480))

    while True:
        ret, frame = video_capture.read()

        out.write(frame)

        cv.imshow("Web Camera", frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    video_capture.release()
    out.release()


def convert_video_to_images(file_name: str, path: str):
    video_capture = cv.VideoCapture(path + file_name)

    count = 0
    while True:
        ret, frame = video_capture.read()
        if frame is None:
            break
        cv.imwrite(path + "images/frame%d.jpg" % count, frame)

        count += 1

        cv.waitKey(30)
