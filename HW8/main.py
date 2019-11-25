""" Homework 8 - Noise Removal
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np
import random
import math
import mathology as mtl


class NoiseImage:
    def __init__(self, img):
        self.img_original = img
        self.img_noise = np.copy(img)
        self.height, self.width = img.shape[:2]
        self.filename_prefix = ''


    # 6 operations after generating noise image
    def operation(self):
        self.noise_box_filter(3)
        self.noise_box_filter(5)
        self.noise_median_filter(3)
        self.noise_median_filter(5)
        self.opening_then_closing()
        self.closing_then_opening()


    # Generate noise images with gaussian noise (amplitude)
    def gaussian_noise(self, amp):
        self.img_noise = np.copy(self.img_original)
        for h in range(self.height):
            for w in range(self.width):
                self.img_noise[h, w] = self.img_noise[h, w] + amp * np.random.normal(0, 1)

        self.filename_prefix = 'gaussian_' + str(amp)
        filename = self.filename_prefix + '.bmp'
        self.write_image(self.img_noise, filename)
        self.count_snr(self.img_noise, filename)

        return self.img_noise


    # Generate noise images with salt and pepper
    def salt_and_pepper_noise(self, p):
        self.img_noise = np.copy(self.img_original)
        for h in range(self.height):
            for w in range(self.width):
                var = random.uniform(0, 1)
                if var < p:
                    self.img_noise[h, w] = 0
                elif var > (1 - p):
                    self.img_noise[h, w] = 255

        self.filename_prefix = 'salt_and_pepper_' + str(p).replace('.', '_')
        filename = self.filename_prefix + '.bmp'
        self.write_image(self.img_noise, filename)
        self.count_snr(self.img_noise, filename)
        
        return self.img_noise


    # Box filter on noise image with size x size
    def noise_box_filter(self, size):
        k = size // 2 # half length of the box width
        img_result = np.copy(self.img_noise)
        for h in range(self.height):
            for w in range(self.width):
                row_start = h - k if h - k >= 0 else 0
                col_start = w - k if w - k >= 0 else 0
                row_end = h + k if h + k < self.height else self.height - 1
                col_end = w + k if w + k < self.width else self.width - 1
                count = total = 0

                for i in range(row_start, row_end + 1):
                    for j in range(col_start, col_end + 1):
                        count += 1
                        total += self.img_noise[i, j]
                img_result[h, w] = total // count

        filename = self.filename_prefix + '_box_' + str(size) + 'x' + str(size) + '.bmp'
        self.write_image(img_result, filename)
        self.count_snr(img_result, filename)

        return img_result


    def noise_median_filter(self, size):
        k = size // 2 # half length of the box width
        img_result = np.copy(self.img_noise)

        for h in range(self.height):
            for w in range(self.width):
                row_start = h - k if h - k >= 0 else 0
                col_start = w - k if w - k >= 0 else 0
                row_end = h + k if h + k < self.height else self.height - 1
                col_end = w + k if w + k < self.width else self.width - 1
                neighbors = []

                for i in range(row_start, row_end + 1):
                    for j in range(col_start, col_end + 1):
                        neighbors.append(self.img_noise[i, j])
                
                neighbors = np.sort(neighbors)
                n = len(neighbors)
                median = neighbors[round((n + 1) / 2) - 1]
                q = neighbors[round((3 * n + 2) / 4) - 1] - neighbors[round((n + 2) / 4) - 1]
                if q == 0:
                    img_result[h, w] = median
                elif abs((int(self.img_noise[h, w]) - median) / q) >= 0.1:
                    img_result[h, w] = median

        filename = self.filename_prefix + '_median_' + str(size) + 'x' + str(size) + '.bmp'
        self.write_image(img_result, filename)
        self.count_snr(img_result, filename)

        return img_result

    
    def opening_then_closing(self):
        img_result = mtl.closing(mtl.opening(self.img_noise))
        filename = self.filename_prefix + '_opening_then_closing.bmp'
        self.write_image(img_result, filename)
        self.count_snr(img_result, filename)

        return img_result


    def closing_then_opening(self):
        img_result = mtl.opening(mtl.closing(self.img_noise))
        filename = self.filename_prefix + '_closing_then_opening.bmp'
        self.write_image(img_result, filename)
        self.count_snr(img_result, filename)

        return img_result


    def count_snr(self, img_noise, filename):
        # Nomalize the images from 0-255 to 0-1
        img = np.copy(self.img_original).astype(np.float) / 255.0
        img_noise = img_noise.astype(np.float) / 255.0

        # Subtraction of two images
        img_diff = img - img_noise
        
        # standar deviation 
        std_s = np.std(img)
        std_n = np.std(img_diff)

        # result formula
        result = round(20 * math.log(std_s / std_n, 10), 3)

        print(filename, '\n  SNR = ' + str(result))

        return result


    def write_image(self, img, filename):
        cv2.imwrite(filename, img)


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    noise = NoiseImage(img)

    noise.gaussian_noise(10)
    noise.operation()

    noise.gaussian_noise(30)
    noise.operation()

    noise.salt_and_pepper_noise(0.1)
    noise.operation()

    noise.salt_and_pepper_noise(0.05)
    noise.operation()


if __name__ == '__main__':
    main()