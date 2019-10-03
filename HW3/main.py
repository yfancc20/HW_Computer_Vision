""" Homework 3 - Histogram Equalization
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# (a)
def original_image(img):
    print('(a) Generating original image and histogram...')
    cv2.imwrite('a.bmp', img)
    histogram_image(img, 'hist-a.png')


# (b)
def darken_image(img, scale):
    print('(b) Dividing the intensity by 3...')
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            img[h, w] = img[h, w] // 3

    cv2.imwrite('b.bmp', img)
    histogram_image(img, 'hist-b.png')
    return img


# (c)
def histogram_equalization(img):
    print('(c) Doing histogram equalization of (b)...')
    height, width = img.shape[:2]
    bins = pixels_count(img)    # rk
    p_r = np.zeros(256)         # probabilities of rk
    s = np.zeros(256, dtype=int)           # sk
    
    # caculate Pr(rk)
    for i in range(len(p_r)):
        p_r[i] = bins[i] / img.size

    # caculate sk
    accu = 0    # accumulation
    for i in range(len(p_r)):
        accu += (255 * p_r[i])
        s[i] = round(accu)

    # T(r) = s
    for h in range(height):
        for w in range(width):
            img[h, w] = s[img[h, w]]

    cv2.imwrite('c.bmp', img)
    histogram_image(img, 'hist-c.png')
    
    
def histogram_image(img, file_name):
    # draw the bar chart of the data (histogram) 
    bins = pixels_count(img)
    plt.clf() # clean the plot
    plt.xlim([0, bins.size])
    plt.bar(np.arange(256), bins, width=1.0, color=(0, 0, 0))
    plt.savefig(file_name)
    # plt.show()


def pixels_count(img):
    height, width = img.shape[:2]
    bins  = np.zeros(256)
    for h in range(height):
        for w in range(width):
            bins[img[h, w]] += 1

    return bins


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    original_image(np.copy(img))
    dark_img = darken_image(np.copy(img), 3)
    histogram_equalization(np.copy(dark_img))


if __name__ == '__main__':
    main()