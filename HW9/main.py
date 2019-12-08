""" Homework 9 - General Edge Detection
Yi-Fan Wu (r08921104)
"""

import cv2
import numpy as np

def roberts_operator(img, threshold):
    height, width = img.shape[:2]
    img_result = np.copy(img)

    for h in range(height):
        for w in range(width):
            right = img[h, w + 1] if w + 1 < width else 0
            bottom = img[h + 1, w] if h + 1 < height else 0
            right_bottom = img[h + 1, w + 1] if w + 1 < width and h + 1 < height else 0

            r1 = right_bottom - img[h, w]
            r2 = bottom - right

    return img_result


def main():
    print('1. Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    img_robert = roberts_operator(img, 30)
    cv2.imwrite('a-robert.bmp', img_robert)
    

if __name__ == '__main__':
    main()