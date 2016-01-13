from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2

def segment(image):
    #threshold with otsu's after meanshift to reduce noise
    img = image
    cv2.imshow("Shifted", img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Threshold", thresh)

    #perform euclidean distance transform
    distances = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(distances, indices = False, min_distance = 3, labels = thresh)

    #perform connected component analysis on local peaks
    markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]
    labels = watershed(-distances, markers, mask = thresh)
    print("[out] {} unique objects found".format(len(np.unique(labels)) - 1))

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
        #cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)

    #show final image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

def main():
    print("Computer vision is hard.")
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    segment(image)

if __name__ == "__main__": main()    
