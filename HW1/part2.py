""" Homework 1 - Basic Image Manipulation - part2
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np


def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (height/2, width/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (height, width))

    cv2.imwrite('part2-d.bmp', rotated_img)


def shrink_image(img):
    height, weight = img.shape[:2]
    dim = (int(height*0.5), int(weight*0.5))
    shrink_img = cv2.resize(img, dim, cv2.INTER_AREA)

    cv2.imwrite('part2-e.bmp', shrink_img)


def binarize_image(img):
    trunc_value = 128
    ret, binarized_img = cv2.threshold(img, trunc_value, 255, cv2.THRESH_TRUNC)
    
    cv2.imwrite('part2-f.bmp', binarized_img)


def main():
    img = cv2.imread('lena.bmp')
    rotate_image(np.copy(img), -45)
    shrink_image(np.copy(img))
    binarize_image(np.copy(img))


main()