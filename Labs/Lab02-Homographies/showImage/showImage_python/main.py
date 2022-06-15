import cv2 as cv

IMAGE_FILE = "../../../Data/sample.jpg"

if __name__ == "__main__":
    img = cv.imread(IMAGE_FILE)
    if img is None:
        print("Error: No image to show")

    cv.imshow("Input image", img)

    # Wait up to 5s for a key press
    cv.waitKey(5000)
