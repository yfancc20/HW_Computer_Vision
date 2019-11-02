""" Homework 5 - Gray Scaled Morphology
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


def dilation(img, se):
    

    return img
    

def erosion(img, se):
    
    return img


def opening(img, se):
    img = dilation(img, se)
    return img


def closing(img, se):
    img = erosion(img, se)
    return img


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    print('(a) Doing dilation...')
    dilation_img = dilation(np.copy(img), SE)
    cv2.imwrite('a-dilation.bmp', dilation_img)

    # print('(b) Doing erosion...')
    # erosion_img = erosion(np.copy(binary_img), SE)
    # cv2.imwrite('b-erosion.bmp', erosion_img)

    # print('(c) Doing opening...')
    # opening_img = opening(np.copy(erosion_img), SE)
    # cv2.imwrite('c-opening.bmp', opening_img)

    # print('(d) Doing closing...')
    # closing_img = closing(np.copy(dilation_img), SE)
    # cv2.imwrite('d-closing.bmp', closing_img)



if __name__ == '__main__':
    main()