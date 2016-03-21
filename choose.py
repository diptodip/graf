import cv2
import numpy as np
import argparse

def adjusted_average(image):
    # blur image
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    # gray image
    blurgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # show blurred image
    #cv2.imshow("Blurred", blurgray)
    mean = cv2.mean(blurgray)[0]
    (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(blurgray)
    print(maxval)
    avgadjust = ((0.45 * (maxval + minval)) + maxval) * 0.5
    return avgadjust

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    ret = adjusted_average(image)
    print(ret)

if __name__ == "__main__": main()
