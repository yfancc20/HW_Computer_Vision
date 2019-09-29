""" Homework 2 - Basic Image Manipulation
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# (a)
def binarize_image(img):
    print('(a) Binarizing...')
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            if img[h][w][0] < 128:
                img[h, w] = (0, 0, 0)
            else:
                img[h, w] = (255, 255, 255)

    cv2.imwrite('a.bmp', img)
    return img


# (b)
def histogram_image(img):
    print('(b) Doing histogram...')
    height, width = img.shape[:2]
    bins  = np.zeros((256))

    for h in range(height):
        for w in range(width):
            bins[img[h, w][0]] += 1
    
    # draw the bar chart of the data (histogram)
      
    plt.bar(np.arange(256), bins)
    plt.show()


# (c)
def connected_components(img):
    print('(c) Finding connected components...')
    height, weight = img.shape[:2]
    


def main():
    print('Reading the image...')
    # img = cv2.imread('lena.bmp')
    # binary_img = binarize_image(np.copy(img))
    # histogram_image(np.copy(img))
    img = cv2.imread('a.bmp')
    connected_components(img)


if __name__ == '__main__':
    main()