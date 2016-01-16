import cv2
import numpy as np
import argparse

def choose(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    mean = cv2.mean(gray)[0]
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("ret: {}".format(ret))
    print("mean: {}".format(mean))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    choose(image)

if __name__ == "__main__": main()
