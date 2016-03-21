import numpy as np
import cv2
import argparse
import choose
import count

def edgy(image):
    # show original image
    cv2.imshow("Image", image)

    # preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)

    # determine adjusted background color and minimum guaranteed
    # brightness color for spots
    minimum = choose.adjusted_average(image)
    maximum = 120

    # perform Canny edge detection
    edges = cv2.Canny(gray, minimum, maximum)

    # perform morphological closing to solidify spots
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # show image after closing
    cv2.imshow("Edge", closed)

    # count spots and show output
    objects = count.segment(image, closed)
    print("[out] Found {} objects.".format(objects))
    cv2.waitKey(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    edgy(image)

if __name__ == "__main__": main()
