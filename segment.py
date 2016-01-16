from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2

def count(image):
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    cv2.imshow("Shifted", shifted)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)

    distances = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(distances, indices = False, min_distance = 20, labels = thresh)
    markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]
    labels = watershed(-distances, markers, mask = thresh)
    print("[out] {} unique objects found".format(len(np.unique(labels)) - 1))

    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(gray.shape, dtype = "uint8")
        mask[labels == label] = 255
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contour = max(contours, key = cv2.contourArea)

        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    cv2.imshow("Output", image)

    cv2.waitKey(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    count(image)

if __name__ == "__main__": main()
