""" Homework 6 - Yokoi Connectivity Number
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np

# a1, a2, a3, a4, 
# and their corresponding neighbors' directions
NEIGHBORS = {
    (0, 1): [(-1, 0), (-1, 1)],     # a1 
    (-1, 0): [(-1, -1), (0, -1)],   # a2
    (0, -1): [(1, -1), (1, 0)],     # a3
    (1, 0): [(1, 1), (0, 1)]        # a4
}



def binarize_image(img):
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            if img[h, w] < 128:
                img[h, w] = 0
            else:
                img[h, w] = 255

    # cv2.imwrite('binary_lena.bmp', img)
    return img


# from 512x512 to 64x64
def downsample(img):
    height, width = img.shape[:2]
    down_img = np.zeros((64, 64), dtype=int)
    for h in range(0, height, 8):
        for w in range(0, width, 8):
            down_img[int(h/8), int(w/8)] = img[h, w]

    cv2.imwrite('downsample.bmp', down_img)
    return down_img


# def count_connectivity(img):



def main():
    print('1. Reading the image and binarizing...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    binary_img = binarize_image(img)

    print('2. Downsampling...')
    down_img = downsample(binary_img)
    print(down_img)

    print('3. Counting Yokoi connectivity...')
    


if __name__ == '__main__':
    main()