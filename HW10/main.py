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
            
            if amount >= threshold:
                img_mask[h-1, w-1] = 1
            elif amount <= (threshold * -1):
                img_mask[h-1, w-1] = -1
            else:
                img_mask[h-1, w-1] = 0

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
            
            if amount >= threshold:
                img_mask[h-1, w-1] = 1
            elif amount <= (threshold * -1):
                img_mask[h-1, w-1] = -1
            else:
                img_mask[h-1, w-1] = 0


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
                
                if not flag:
                    img_result[h-1, w-1] = 255
    
    return img_result


def check_threshold(amount):
    if amount >= threshold:
        return 1
    elif amount <= (threshold * -1):
        return -1
    else:
        return 0


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    # laplacian(img, 15)
    minimum_variance_laplacian(img, 20)
    

if __name__ == '__main__':
    main()