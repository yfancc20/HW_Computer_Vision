""" Homework 8 - Noise Removal
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np
import random
import math
import mathology as mtl


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
    img_result = np.copy(img)
    for h in range(height):
        for w in range(width):
            var = random.uniform(0, 1)
            if var < p:
                img_result[h, w] = 0
            elif var > (1 - p):
                img_result[h, w] = 255
    
    return img_result


def noise_box_filter(img, size):
    height, width = img.shape[:2]
    k = size // 2 # half length of the box width
    img_result = np.copy(img)
    for h in range(height):
        for w in range(width):
            row_start = h - k if h - k >= 0 else 0
            col_start = w - k if w - k >= 0 else 0
            row_end = h + k if h + k < height else height - 1
            col_end = w + k if w + k < width else width - 1
            count = total = 0

            for i in range(row_start, row_end + 1):
                for j in range(col_start, col_end + 1):
                    count += 1
                    total += img[i, j]
            img_result[h, w] = total // count

    return img_result


def noise_median_filter(img, size):
    height, width = img.shape[:2]
    k = size // 2 # half length of the box width
    img_result = np.copy(img)

    for h in range(height):
        for w in range(width):
            row_start = h - k if h - k >= 0 else 0
            col_start = w - k if w - k >= 0 else 0
            row_end = h + k if h + k < height else height - 1
            col_end = w + k if w + k < width else width - 1
            neighbors = []

            for i in range(row_start, row_end + 1):
                for j in range(col_start, col_end + 1):
                    neighbors.append(img[i, j])
            
            neighbors = np.sort(neighbors)
            n = len(neighbors)
            median = neighbors[round((n + 1) / 2) - 1]
            q = neighbors[round((3 * n + 2) / 4) - 1] - neighbors[round((n + 2) / 4) - 1]
            if q == 0:
                img_result[h, w] = median
            elif abs((int(img[h, w]) - median) / q) >= 0.1:
                img_result[h, w] = median

    return img_result



def count_snr(img, img_noise):
    # Nomalize the images from 0-255 to 0-1
    img = img.astype(np.float) / 255.0
    img_noise = img_noise.astype(np.float) / 255.0

    # Subtraction of two images
    img_diff = img - img_noise
    
    # standar deviation 
    std_s = np.std(img)
    std_n = np.std(img_diff)

    # result formula
    result = round(20 * math.log(std_s / std_n, 10), 3)

    return result





def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    # print('Generating gaussian noise images with amptitude 30...')
    # noise_img = gaussian_noise(np.copy(img), 30)
    # cv2.imwrite('a-gaussian_noise_30.bmp', noise_img)

    print('Generating salt and pepper noise images with probability 0.1...')
    noise_img = salt_and_pepper_noise(img, 0.1)
    cv2.imwrite('b-salt_and_pepper_noise_0_1.bmp', noise_img)
    # snr = count_snr(img, noise_img)
    # print(snr)
    
    # removal_img = noise_box_filter(noise_img, 3)
    # cv2.imwrite('c-removal.bmp', removal_img)

    # removal_img = noise_median_filter(noise_img, 5)
    # cv2.imwrite('d-removal.bmp', removal_img)

    # snr = count_snr(img, removal_img)
    # print(snr)

    cto_img = mtl.closing_then_opening(noise_img)
    cv2.imwrite('e-otc.bmp', cto_img)
    snr = count_snr(img, cto_img)
    print(snr)



if __name__ == '__main__':
    main()