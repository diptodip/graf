import cv2
import numpy as np
import argparse

def choose(image):
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    blurgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    mean = cv2.mean(gray)[0]
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(blurgray)
    cv2.circle(image, maxloc, 5, (255, 0, 0), 2)
    naiveadjust = 1.3 * mean
    avgadjust = (maxval + mean)/2
    #print("ret: {}".format(ret))
    #print("mean: {}".format(mean))
    #print("naive: {}".format(naiveadjust))
    #print("avg adjust: {}".format(avgadjust))
    #cv2.imshow("Brightest Spot", image)
    #cv2.imshow("Blurred", blurgray)
    #cv2.waitKey(0)
    return avgadjust

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    choose(image)

if __name__ == "__main__": main()
