""" Homework 7 - Thinning
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


def thinning(img):
    yokoi_matrix = count_connectivity(img)
    print(yokoi_matrix)


def count_connectivity(img):
    height, width = img.shape[:2]
    # result will be a 64x64 matrix
    result = []
    
    for h in range(height):
        row_arr = []
        for w in range(width):
            count = 0
            if img[h, w]:
                # 4 neighbors need to be checked
                count_r = 0
                for n, d in NEIGHBORS.items():
                    if 0 <= h + n[0] < height and 0 <= w + n[1] < width:
                        neighbor = img[h + n[0], w + n[1]]
                        if neighbor:
                            h1, w1 = h + d[0][0], w + d[0][1]
                            h2, w2 = h + d[1][0], w + d[1][1]
                            # two case for counting:
                            if h1 < 0 or h1 == height or w1 < 0 or w1 == width or \
                                h2 < 0 or h2 == height or w2 < 0 or w2 == width:
                                # 1. out of bound
                                count += 1
                            elif img[h1, w1] != neighbor or img[h2, w2] != neighbor:
                                # 2. one of pixel not equal
                                count += 1 
                            elif img[h1, w1] == neighbor and img[h2, w2] == neighbor:
                                count_r += 1
                if count_r == 4:
                    count = 5

            if count == 0:
                row_arr.append(' ')
            else:
                row_arr.append(count)
        result.append(row_arr)

    return result




def main():
    print('1. Reading the image and binarizing...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    binary_img = binarize_image(img)

    print('2. Downsampling...')
    down_img = downsample(binary_img)

    print('3. Thinning...')
    thin_img = thinning(down_img)
    


if __name__ == '__main__':
    main()