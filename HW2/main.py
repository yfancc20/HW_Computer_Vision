""" Homework 2 - Basic Image Manipulation
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np


# (a)
def binarize_image(img):
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            if img[h][w][0] < 128:
                img[h, w] = (0, 0, 0)
            else:
                img[h, w] = (255, 255, 255)

    cv2.imwrite('a.bmp', img)


def main():
    img = cv2.imread('lena.bmp')
    binarize_image(np.copy(img))

if __name__ == '__main__':
    main()