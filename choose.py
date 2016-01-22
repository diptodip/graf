import cv2
import numpy as np
import argparse

def choose(image):
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(gray)[0]
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    adjust = 1.3 * mean
    print("ret: {}".format(ret))
    print("mean: {}".format(mean))
    print("adjust: {}".format(adjust))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    choose(image)

if __name__ == "__main__": main()
