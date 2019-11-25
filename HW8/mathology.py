""" Homework 8 
This file includes the functions that previous HW used.

Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np


SE = [
    (-2, -1), (-2, 0), (-2, 1),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
    (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
    (2, -1), (2, 0), (2, 1)
] # structure element (3-5-5-5-3 kernel)


def dilation(img):
    height, width = img.shape
    img_res = np.copy(img)

    for h in range(height):
        for w in range(width):
            maxima = 0
            for pixel in SE:
                x, y = pixel
                if 0 <= h + x < height and 0 <= w + y < width:
                    maxima = max(maxima, img[h+x, w+y])
            
            img_res[h, w] = maxima

    return img_res
    

def erosion(img):
    height, width = img.shape
    img_res = np.copy(img)

    for h in range(height):
        for w in range(width):
            # only concern the case that se with origin here
            minimum = np.inf
            for pixel in SE:
                x, y = pixel
                if 0 <= h + x < height and 0 <= w + y < width:
                    minimum = min(minimum, img[h+x, w+y])

            img_res[h, w] = minimum
        
    return img_res


def opening(img):
    img_res = dilation(erosion(img))
    return img_res


def closing(img):
    img_res = erosion(dilation(img))
    return img_res