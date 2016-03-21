import cv2
import numpy as np
import argparse

def adjusted_average(image, preprocessed):
    # preprocess image
    if not preprocessed:
        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        blurgray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    else:
        blurgray = cv2.GaussianBlur(image, (9, 9), 0)

    # find mean and adjust value using max value
    mean = cv2.mean(blurgray)[0]
    (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(blurgray)
    avgadjust = ((0.45 * (maxval + minval)) + maxval) * 0.5
    return avgadjust

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    ret = adjusted_average(image, False)
    print(ret)

if __name__ == "__main__": main()
