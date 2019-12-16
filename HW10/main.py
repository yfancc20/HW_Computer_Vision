""" Homework 10 - Zero Crossing Edge Detection
Yi-Fan Wu (r08921104)
"""

import cv2
import numpy as np
import math


def laplacian(img, threshold):
    print('Laplacian Mask x2 with threshold ' + str(threshold))

    height, width = img.shape[:2]
    img_mask = np.zeros(img.shape[:2], dtype=int)

    # Replicate and padding the border
    img_rep = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # Fisrt Mask
    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    for h in range(1, height + 1):
        for w in range(1, width + 1):
            amount = -4 * img_rep[h, w]
            for x, y in neighbors:
                amount += img_rep[h+x, w+y]
            
            if amount >= threshold:
                img_mask[h-1, w-1] = 1
            elif amount <= (threshold * -1):
                img_mask[h-1, w-1] = -1
            else:
                img_mask[h-1, w-1] = 0

    img_mask_rep = cv2.copyMakeBorder(img_mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    img_result = zero_crossing(img_mask_rep)
    cv2.imwrite('a-laplacian_1.bmp', img_result)

    # Second Mask
    for h in range(1, height + 1):
        for w in range(1, width + 1):
            amount = 0
            for x in range(h - 1, h + 2):
                for y in range(w - 1, w + 2):
                    amount += img_rep[x, y]
            amount -= 9 * img_rep[h, w]
            amount /= 3

            img_mask[h-1, w-1] = check_threshold(amount, threshold)

    img_mask_rep = cv2.copyMakeBorder(img_mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    img_result = zero_crossing(img_mask_rep)
    cv2.imwrite('a-laplacian_2.bmp', img_result)

    return img_result


def minimum_variance_laplacian(img, threshold):
    print('Minumum-variance laplacian mask with threshold ' + str(threshold))

    height, width = img.shape[:2]
    img_mask = np.zeros(img.shape[:2], dtype=int)

    # Replicate and padding the border
    img_rep = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    neighbors_1 = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    neighbors_2 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for h in range(1, height + 1):
        for w in range(1, width + 1):
            amount = -4 * img_rep[h, w]
            for x, y in neighbors_1:
                amount -= img_rep[h+x, w+y]
            for x, y in neighbors_2:
                amount += 2 * img_rep[h+x, w+y]
            amount /= 3
            
            img_mask[h-1, w-1] = check_threshold(amount, threshold)

    img_mask_rep = cv2.copyMakeBorder(img_mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    img_result = zero_crossing(img_mask_rep)
    cv2.imwrite('b-minumum_varaince.bmp', img_result)

    return img_result


def laplacian_gaussian(img, threshold):
    print('Laplacian gaussain mask with threshold ' + str(threshold))

    matrix = [
        0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0,
        0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0,
        0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0,
        -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1,
        -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1,
        -2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2,
        -1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1,
        -1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1,
        0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0,
        0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0,
        0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0
    ]

    height, width = img.shape[:2]
    img_mask = np.zeros(img.shape[:2], dtype=int)

    # Replicate and padding the border
    img_rep = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)

    for h in range(5, height + 5):
        for w in range(5, width + 5):
            amount = i = 0
            for x in range(h - 5, h + 6):
                for y in range(w - 5, w + 6):
                    amount += img_rep[x, y] * matrix[i]
                    i += 1
            
            img_mask[h-5, w-5] = check_threshold(amount, threshold)

    img_mask_rep = cv2.copyMakeBorder(img_mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    img_result = zero_crossing(img_mask_rep)
    cv2.imwrite('c-laplacian_gaussian.bmp', img_result)

    return img_result


def laplacian_difference_gaussian(img, threshold):
    print('Laplacian gaussain mask with threshold ' + str(threshold))

    matrix = [
        -1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1,
        -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3,
        -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4,
        -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6,
        -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7,
        -8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8,
        -7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7,
        -6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6,
        -4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4,
        -3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3,
        -1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1
    ]

    img_result = gaussian_operation(img, threshold, matrix)
    cv2.imwrite('d-laplacian_difference_gaussian.bmp', img_result)

    return img_result


def gaussian_operation(img, threshold, matrix):
    height, width = img.shape[:2]
    img_mask = np.zeros(img.shape[:2], dtype=int)

    # Replicate and padding the border
    img_rep = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REPLICATE)

    for h in range(5, height + 5):
        for w in range(5, width + 5):
            amount = i = 0
            for x in range(h - 5, h + 6):
                for y in range(w - 5, w + 6):
                    amount += img_rep[x, y] * matrix[i]
                    i += 1
            
            img_mask[h-5, w-5] = check_threshold(amount, threshold)

    img_mask_rep = cv2.copyMakeBorder(img_mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    
    return zero_crossing(img_mask_rep)


# The input image have been padded (514x514)
def zero_crossing(img):
    height, width = img.shape[:2]
    img_result = np.zeros((512, 512), dtype=np.uint8)

    for h in range(1, height - 1):
        for w in range(1, width - 1):
            if img[h, w] < 1:
                img_result[h-1, w-1] = 255
            else:
                flag = False # Test if -1 surrounding
                for x in range(h - 1, h + 2):
                    for y in range(w - 1, w + 2):
                        if img[x, y] == -1 and\
                             (x != h) and (y != w):
                             flag = True
                             break
                    if flag:
                        break
                
                if not flag:
                    img_result[h-1, w-1] = 255
    
    return img_result


def check_threshold(amount, threshold):
    if amount >= threshold:
        return 1
    elif amount <= (threshold * -1):
        return -1
    else:
        return 0


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    laplacian(img, 15)
    minimum_variance_laplacian(img, 20)
    laplacian_gaussian(img, 3000)
    laplacian_difference_gaussian(img, 1)
    

if __name__ == '__main__':
    main()