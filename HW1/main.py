""" Homework 1 - Basic Image Manipulation
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np


# (a)
def up_down_image(img):
    height, width = img.shape[:2]
    height_end = height - 1

    # swap(1st, nth), swap(2nd, n-1th)...
    for h in range(height//2):
        img[[h]], img[[height_end]] = img[[height_end]], img[[h]]
        height_end -= 1

    cv2.imwrite('part1-a.bmp', img)


# (b)
def right_left_image(img):
    height, width = img.shape[:2]
    
    for h in range(height):
        width_end = width - 1
        for w in range(width//2):
            img[h, [w, width_end]] = img[h, [width_end, w]]
            width_end -= 1
    
    cv2.imwrite('part1-b.bmp', img)


# (c)
def diagonally_mirrored_image(img):
    # assume the image is square, from top-left to bottom-right
    height, width = img.shape[:2]
    height_end = 0
    
    for h in range(height):
        for w in range(height_end):
            img[h, w] = img[w, h]
        height_end += 1

    cv2.imwrite('part1-c.bmp', img)


# (d)
def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (height/2, width/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (height, width))

    cv2.imwrite('part2-d.bmp', rotated_img)


# (e)
def shrink_image(img):
    height, weight = img.shape[:2]
    dim = (int(height*0.5), int(weight*0.5))
    shrink_img = cv2.resize(img, dim, cv2.INTER_AREA)

    cv2.imwrite('part2-e.bmp', shrink_img)


# (f)
def binarize_image(img):
    trunc_value = 128
    ret, binarized_img = cv2.threshold(img, trunc_value, 255, cv2.THRESH_TRUNC)
    
    cv2.imwrite('part2-f.bmp', binarized_img)


def main():
    img = cv2.imread('lena.bmp')
    up_down_image(np.copy(img))
    right_left_image(np.copy(img))
    diagonally_mirrored_image(np.copy(img))
    rotate_image(np.copy(img), -45)
    shrink_image(np.copy(img))
    binarize_image(np.copy(img))


main()