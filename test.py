import numpy as np
import cv2

img = cv2.imread('coins.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
out, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #remove noise
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    #find sure bg
surebg = cv2.dilate(opening, kernel, iterations = 3)

    #find sure fg
distanceTransform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
out, surefg = cv2.threshold(distanceTransform, 0.7 * distanceTransform.max(), 255, 0)

    #find unsure area
surefg = np.uint8(surefg)
unsure = cv2.subtract(surebg, surefg)

    #label markers
out, markers = cv2.connectedComponents(surefg)

    #make surebg 1 instead of 0
markers = markers + 1

    #mark unknowns as 0
markers[unsure==255] = 0

markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

count = 0

for marker in np.unique(markers):
    if marker == 0:
        continue
    count = count + 1

print(count)

cv2.imshow('img', img)
cv2.imshow('markers', markers)
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
