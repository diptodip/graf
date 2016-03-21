from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2
import choose

def segment(image, thresh):
    #threshold with otsu's after meanshift to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #perform euclidean distance transform
    distances = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(distances, indices = False, min_distance = 3, labels = thresh)

    #perform connected component analysis on local peaks
    markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]
    labels = watershed(-distances, markers, mask = thresh)

    #loop over labels returned from watershed to mark them
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        #find contours in mask and choose biggest contour by area
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contour = max(contours, key = cv2.contourArea)

        #draw circle around max size contour
        ((x, y), r) = cv2.minEnclosingCircle(contour)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    
    #show final image
    cv2.imshow("Output", image)
    return len(np.unique(labels) - 1)

def main():
    print("Computer vision is hard.")
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    out = segment(image)

if __name__ == "__main__": main() 
