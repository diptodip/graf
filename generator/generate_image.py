from random import randint
from statistics import median
from math import sqrt

import numpy as np
from tifffile import imsave

GRID_SIZE = 32
AVERAGE_BRIGHTNESS = 2500
STDDEV = 250

def generate_image(file_prefix = 'test'):
    NUM_SPOTS = randint(25, 40)
    NUM_FRAMES = randint(18, 32)
    I = np.random.normal(AVERAGE_BRIGHTNESS, STDDEV, (NUM_FRAMES, GRID_SIZE, GRID_SIZE)).astype(np.uint16)
    I_ = np.zeros((NUM_FRAMES, GRID_SIZE, GRID_SIZE)).astype(np.uint16)
    for spot in range(NUM_SPOTS):
        generate_spot(I, I_, NUM_FRAMES)
    I = np.reshape(I, (NUM_FRAMES, GRID_SIZE, GRID_SIZE), order='A')
    print(I.shape)
    print(NUM_SPOTS)
    imsave(file_prefix + '.tiff', I)
    imsave(file_prefix + '_label.tiff', I_)

def generate_spot(I, I_, NUM_FRAMES):
    max_brightness = 0
    while not (max_brightness > 3000):
        max_brightness = int(round(np.random.normal(2.5 * AVERAGE_BRIGHTNESS, STDDEV)))
    max_radius_x = int(round(np.random.normal(2, 1.0)))
    max_radius_y = int(round(np.random.normal(2, 1.0)))
    while not (max_radius_x > 1 and max_radius_y > 1):
        max_radius_x = int(round(np.random.normal(2, 1.5)))
        max_radius_y = int(round(np.random.normal(2, 1.5)))
    x = abs(min(int(round(np.random.normal(GRID_SIZE / 2.0, 0.3 * GRID_SIZE))), GRID_SIZE - 1))
    y = abs(min(int(round(np.random.normal(GRID_SIZE / 2.0, 0.3 * GRID_SIZE))), GRID_SIZE - 1))
    z = randint(0, NUM_FRAMES - 1)
    print((x, y, z))
    frame_radius = randint(9, 16)
    max_distance = sqrt(max_radius_x**2 + max_radius_y**2)
    I_[z, x, y]  = 6000
    for i in range(-frame_radius, frame_radius):
        for j in range(-max_radius_x, max_radius_x):
            for k in range(-max_radius_y, max_radius_y):
                a = x + j
                if a < 0: a = 0
                if a >= GRID_SIZE: a = GRID_SIZE - 1
                b = y + k
                if b < 0: b = 0
                if b >= GRID_SIZE: b = GRID_SIZE - 1
                if z + i > NUM_FRAMES - 1:
                    break
                distance = sqrt(j**2 + k**2)
                brightness = randint(int((max_brightness - AVERAGE_BRIGHTNESS) / 2), max_brightness - AVERAGE_BRIGHTNESS)
                brightness *= (1 - distance/max_distance)
                brightness *= (1 - abs(i) / frame_radius)
                I[z + i, a, b] += brightness

if __name__=='__main__': generate_image()
