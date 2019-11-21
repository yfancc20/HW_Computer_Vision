""" Homework 8 - Noise Removal
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np
import random


# Generate noise images with gaussian noise (amplitude)
def gaussian_noise(img, amp):
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            img[h, w] = img[h, w] + amp * np.random.normal(0, 1)

    return img

# Generate noise images with salt and pepper
def salt_and_pepper_noise(img, p):
    height, width = img.shape[:2]
    for h in range(height):
        for w in range(width):
            var = random.uniform(0, 1)
            if var < p:
                img[h, w] = 0
            elif var > (1 - p):
                img[h, w] = 255
    
    return img


def noise_removal_with_box(img, size):


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    print('Generating gaussian noise images with amptitude 30...')
    noise_img = gaussian_noise(np.copy(img), 30)
    cv2.imwrite('a-gaussian_noise_30.bmp', noise_img)

    print('Generating salt and pepper noise images with probability 0.1...')
    noise_img = salt_and_pepper_noise(np.copy(img), 0.1)
    cv2.imwrite('b-salt_and_pepper_noise_0_1.bmp', noise_img)
    


if __name__ == '__main__':
    main()