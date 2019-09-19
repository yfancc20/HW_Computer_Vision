""" Homework 1 - Basic Image Manipulation - part1
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np


def up_down_image(img):
    row, col = img.shape[:2]
    row_end = row - 1

    # swap(1st, nth), swap(2nd, n-1th)...
    for x in range(row//2):
        img[[x]], img[[row_end]] = img[[row_end]], img[[x]]
        row_end -= 1

    cv2.imwrite('part1-a.bmp', img)


def right_left_image(img):
    row, col = img.shape[:2]
    
    for r in range(row):
        col_end = col - 1
        for c in range(col//2):
            img[r, [c, col_end]] = img[r, [col_end, c]]
            col_end -= 1
    
    cv2.imwrite('part1-b.bmp', img)


def diagonally_mirrored_image(img):
    # assume the image is square, from top-left to bottom-right
    row, col = img.shape[:2]
    row_end = 0
    
    for r in range(row):
        for c in range(row_end):
            img[r, c] = img[c, r]
        row_end += 1

    cv2.imwrite('part1-c.bmp', img)


def main():
    img = cv2.imread('lena.bmp')
    up_down_image(np.copy(img))
    right_left_image(np.copy(img))
    diagonally_mirrored_image(np.copy(img))


main()