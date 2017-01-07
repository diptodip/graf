from random import randint
from statistics import median
from math import sqrt

import numpy as np

GRID_SIZE = 32
AVERAGE_BRIGHTNESS = 2000

def generate_image():
    NUM_FRAMES = randint(40, 90)
    NUM_SPOTS = randint(0, 200)
    I = np.zeros((GRID_SIZE, GRID_SIZE, NUM_FRAMES), np.uint16)
    for spot in range(NUM_SPOTS):
        max_brightness = randint(3000, 7000)
        max_radius = randint(1, 4)
        x = randint(0, GRID_SIZE - 1)
        y = randint(0, GRID_SIZE - 1)
        frame_range = randint(5, 12)
        median_frame = int(round(median(range(frame_range))))
        for i in range(frame_range):
            generate_spot(I, x, y, max_radius, max_brightness, median_frame, i)
    print(I)

def generate_spot(I, x, y, max_radius, max_brightness, median_frame, frame_index):
    for i in range(-max_radius, max_radius):
        for j in range(-max_radius, max_radius):
            a = x + i
            if a < 0: a = 0
            if a >= GRID_SIZE: a = GRID_SIZE - 1
            b = y + j
            if b < 0: b = 0
            if b >= GRID_SIZE: b = GRID_SIZE - 1
            distance = sqrt(i**2 + j**2)
            brightness = randint(0, max_brightness - AVERAGE_BRIGHTNESS)
            brightness = int(round(brightness * (1 - float(distance) / max_radius) * abs(median_frame - frame_index))) + AVERAGE_BRIGHTNESS
            I[a, b, frame_index] = brightness

if __name__=='__main__': generate_image()
