""" Homework 8 - Noise Removal
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np


# Generate noise images with gaussian noise (amplitude)
def gaussian_noise(amp):
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            img[h, w] = img[h, w] + amp * np.random.normal(0, 1)

    return img


def main():
    print('1. Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    noise_img = gaussian_noise(np.copy(img), 10)
    cv2.imwrite('a-gaussian_noise_10', noise_img)
    noise_img = gaussian_noise(np.copy(img), 30)
    cv2.imwrite('a-gaussian_noise_30', noise_img)

    print('2. Downsampling...')
    down_img = downsample(binary_img)

    print('3. Thinning...')
    thin_img = thinning(down_img)
    cv2.imwrite('thinning.bmp', thin_img)
    


if __name__ == '__main__':
    main()