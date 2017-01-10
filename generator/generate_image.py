from random import randint
from statistics import median
from math import sqrt

import numpy as np
from tifffile import imsave

GRID_SIZE = 32
AVERAGE_BRIGHTNESS = 2000
STDDEV = 50

def generate_image():
    NUM_SPOTS = randint(5, 90)
    NUM_FRAMES = randint(40, 90)
    I = np.random.normal(AVERAGE_BRIGHTNESS, STDDEV, (NUM_FRAMES, GRID_SIZE, GRID_SIZE)).astype(np.uint16)
    for spot in range(NUM_SPOTS):
        generate_spot(I, NUM_FRAMES)
    I = np.reshape(I, (NUM_FRAMES, GRID_SIZE, GRID_SIZE), order='A')
    print(I.shape)
    imsave('test.tiff', I)

def generate_spot(I, NUM_FRAMES):
    max_brightness = randint(2500, 4000)
    max_radius_x = randint(1, 3)
    max_radius_y = randint(1, 3)
    x = randint(0, GRID_SIZE - 1)
    y = randint(0, GRID_SIZE - 1)
    z = randint(0, NUM_FRAMES - 1)
    frame_range = randint(5, 12)
    median_frame = int(round(median(range(frame_range))))
    for i in range(frame_range):
        for j in range(-max_radius_x, max_radius_x):
            for k in range(-max_radius_y, max_radius_y):
                a = x + j
                if a < 0: a = 0
                if a >= GRID_SIZE: a = GRID_SIZE - 1
                b = y + k
                if b < 0: b = 0
                if b >= GRID_SIZE: b = GRID_SIZE - 1
                if z + i >= NUM_FRAMES - 1:
                    break
                distance = sqrt(j**2 + k**2)
                brightness = randint(0, max_brightness - AVERAGE_BRIGHTNESS)
                brightness = int(round(brightness * (1 - float(distance) / max_radius_x) * (1 - abs(median_frame - i) / float(median_frame)))) + AVERAGE_BRIGHTNESS
                I[z + i, a, b] = brightness

if __name__=='__main__': generate_image()
