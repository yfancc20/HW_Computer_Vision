""" Homework 10 - Zero Crossing Edge Detection
Yi-Fan Wu (r08921104)
"""

import cv2
import numpy as np
import math
  

# Extend the image by ? distance
def extend_image(img, dist=1):
    height, width = img.shape[:2]
    img_extend = np.zeros((height + dist*2, width + dist*2), dtype=int)
    for h in range(height):
        for w in range(width):
            img_extend[h + dist, w + dist] = img[h, w]
            
    # 4 corners of the extended image
    for i in range(0, dist):
        for j in range(0, dist):
            # top-left
            img_extend[i, j] = img_extend[dist, dist]
            # top-right
            img_extend[i, width + dist*2 - 1 - j] = img_extend[dist, width + dist - 1]
            # bottom-left
            img_extend[height + dist*2 - 1 - i, j] = img_extend[height + dist - 1, dist]
            # bottom-right
            img_extend[height + dist*2 - 1 - i, width + dist*2 - 1 - j]\
                = img_extend[height + dist - 1, width + dist - 1]

    # 4 edges of the extended image
    for c in range(dist, width + dist):
        for r in range(0, dist):
            img_extend[r, c] = img_extend[dist, c]
            img_extend[height + dist + r, c] = img_extend[height + dist - 1, c]
    for r in range(dist, height + dist):
        for c in range(0, dist):
            img_extend[r, c] = img_extend[r, dist]
            img_extend[r, width + dist + c] = img_extend[r, width + dist - 1]

    return img_extend


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    

if __name__ == '__main__':
    main()