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
    height, width = img.shape
    img_res = np.zeros(img.shape, dtype=int)

    for h in range(height):
        for w in range(width):
            if img[h, w] > 0: # value != 0
                maxima = 0
                for pixel in se:
                    x, y = pixel
                    if 0 <= h + x < height and 0 <= w + y < width:
                        if img[h+x, w+y] > maxima:
                            maxima = img[h+x, w+y]
                
                for pixel in se:
                    x, y = pixel
                    if 0 <= h + x < height and 0 <= w + y < width:
                        img_res[h+x, w+y] = maxima

    return img_res
    

def erosion(img, se):
    height, width = img.shape
    img_res = np.zeros(img.shape, dtype=int)

    for h in range(height):
        for w in range(width):
            # only concern the case that se with origin here
            if img[h, w] > 0: 
                minimum = np.inf
                all_contained = True
                for pixel in se:
                    x, y = pixel
                    if 0 <= h + x < height and 0 <= w + y < width:
                        if img[h+x, w+y] == 0:
                            all_contained = False
                            break
                        if img[h+x, w+y] < minimum:
                            minimum = img[h+x, w+y]
                
                if all_contained:
                    for pixel in se:
                        x, y = pixel
                        if 0 <= h + x < height and 0 <= w + y < width:
                            img_res[h+x, w+y] = minimum
        
    return img_res


def opening(img, se):
    img_res = dilation(img, se)
    return img_res


def closing(img, se):
    img_res = erosion(img, se)
    return img_res


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    print('(a) Doing dilation...')
    img_dilation = dilation(np.copy(img), SE)
    cv2.imwrite('a-dilation.bmp', img_dilation)

    print('(b) Doing erosion...')
    img_erosion = erosion(np.copy(img), SE)
    cv2.imwrite('b-erosion.bmp', img_erosion)

    print('(c) Doing opening...')
    img_opening = opening(np.copy(img_erosion), SE)
    cv2.imwrite('c-opening.bmp', img_opening)

    print('(d) Doing closing...')
    img_closing = closing(np.copy(img_dilation), SE)
    cv2.imwrite('d-closing.bmp', img_closing)



if __name__ == '__main__':
    main()