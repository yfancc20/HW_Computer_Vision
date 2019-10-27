""" Homework 4 - Binary Morphology
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

SE_J = [(0, -1), (0, 0), (1, 0)]
SE_K = [(-1, 0), (-1, 1), (0, 1)]

def binarize_image(img):
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            if img[h, w] < 128:
                img[h, w] = 0
            else:
                img[h, w] = 255

    cv2.imwrite('binary_lena.bmp', img)
    return img

def dilation(img, se):
    height, width = img.shape[:2]
    added = []
    
    for h in range(height):
        for w in range(width):
            # only black pixels can be added
            if img[h, w] == 0:
                for pixel in se:
                    check_x = h + pixel[0]
                    check_y = w + pixel[1]
                    if 0 <= check_x < height and 0 <= check_y < width: # check boundary
                        if img[check_x, check_y] == 255: # hit the white pixel
                            added.append((h, w)) # be added
                            break

    for pixel in added:
        img[pixel[0], pixel[1]] = 255

    return img
    

def erosion(img, se):
    height, width = img.shape[:2]
    erased = []
    added = []
    
    for h in range(height):
        for w in range(width):
            for pixel in se:
                check_x = h + pixel[0]
                check_y = w + pixel[1]
                all_contained = True
                if 0 <= check_x < height and 0 <= check_y < width: # check boundary
                    if img[check_x, check_y] == 0: # one neighbor is black 
                        all_contained = False
                        break
            
            if not all_contained:
                erased.append((h, w))
            elif img[h, w] == 0:
                added.append((h, w))
    
    for pixel in erased:
        img[pixel[0], pixel[1]] = 0

    for pixel in added:
        img[pixel[0], pixel[1]] = 255
    
    return img


def opening(img, se):
    img = dilation(img, se)
    return img


def closing(img, se):
    img = erosion(img, se)
    return img


def hit_and_miss(img, se_j, se_k):
    height, width = img.shape[:2]

    # get the complement of the image
    img_c = np.copy(img)
    for h in range(height):
        for w in range(width):
            img_c[h, w] = (img_c[h, w] + 255) % 510

    # doing 2 erosions
    img_1 = erosion(np.copy(img), se_j)
    img_2 = erosion(np.copy(img_c), se_k)

    # img_1 and img_2
    for h in range(height):
        for w in range(width):
            if img_1[h, w] == 255 and img_2[h, w] == 255:
                img[h, w] = 255
            else:
                img[h, w] = 0

    return img


def main():
    print('Reading the image and binarizing...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    binary_img = binarize_image(np.copy(img))

    print('(a) Doing dilation...')
    dilation_img = dilation(np.copy(binary_img), SE)
    cv2.imwrite('a-dilation.bmp', dilation_img)

    print('(b) Doing erosion...')
    erosion_img = erosion(np.copy(binary_img), SE)
    cv2.imwrite('b-erosion.bmp', erosion_img)

    print('(c) Doing opening...')
    opening_img = opening(np.copy(erosion_img), SE)
    cv2.imwrite('c-opening.bmp', opening_img)

    print('(d) Doing closing...')
    closing_img = closing(np.copy(dilation_img), SE)
    cv2.imwrite('d-closing.bmp', closing_img)

    print('(e) Doing hit-and-miss...')
    hms_img = hit_and_miss(np.copy(binary_img), SE_J, SE_K)
    cv2.imwrite('e-hit_and_miss.bmp', hms_img)


if __name__ == '__main__':
    main()