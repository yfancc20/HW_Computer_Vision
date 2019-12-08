""" Homework 9 - General Edge Detection
Yi-Fan Wu (r08921104)
"""

import cv2
import numpy as np
import math

def roberts_operator(img, threshold):
    print('Robert\'s edge detection with threshold ' + str(threshold))
    height, width = img.shape[:2]
    img_result = np.copy(img)

    for h in range(height):
        for w in range(width):
            right = int(img[h, w + 1]) if w + 1 < width else 0
            bottom = int(img[h + 1, w]) if h + 1 < height else 0
            right_bottom = int(img[h + 1, w + 1]) if w + 1 < width and h + 1 < height else 0

            r1 = right_bottom - int(img[h, w])
            r2 = bottom - right
            mag = math.sqrt(r1 * r1 + r2 * r2)

            if mag >= threshold:
                img_result[h, w] = 0
            else:
                img_result[h, w] = 255

    cv2.imwrite('a-robert.bmp', img_result)

    return img_result


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    roberts_operator(img, 30)
    

if __name__ == '__main__':
    main()