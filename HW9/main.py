""" Homework 9 - General Edge Detection
Yi-Fan Wu (r08921104)
"""

import cv2
import numpy as np
import math

def robert_operator(img, threshold):
    print('Robert\'s edge detection with threshold ' + str(threshold))

    height, width = img.shape[:2]
    img_result = np.zeros(img.shape[:2], dtype=np.uint8)

    for h in range(height):
        for w in range(width):
            right = int(img[h, w+1]) if w + 1 < width else 0
            bottom = int(img[h+1, w]) if h + 1 < height else 0
            right_bottom = int(img[h+1, w+1]) if w + 1 < width and h + 1 < height else 0

            r1 = right_bottom - int(img[h, w])
            r2 = bottom - right
            mag = math.sqrt(r1*r1 + r2*r2)

            if mag < threshold:
                img_result[h, w] = 255 # else will be zero

    cv2.imwrite('a-robert.bmp', img_result)

    return img_result


def prewitt_operator(img, threshold):
    print('Prewitt\'s edge detection with threshold ' + str(threshold))

    height, width = img.shape[:2]
    img_result = np.zeros(img.shape[:2], dtype=np.uint8)

    # Extend the image
    # The height & width will be the original's plus 2
    img_extend = extend_image(img)

    for h in range(1, height + 1):
        for w in range(1, width + 1):
            p1 = (img_extend[h+1, w-1] + img_extend[h+1, w] + img_extend[h+1, w+1] 
                 - img_extend[h-1, w-1] - img_extend[h-1, w] - img_extend[h-1, w+1])
            p2 = (img_extend[h-1, w+1] + img_extend[h, w+1] + img_extend[h+1, w+1]
                 - img_extend[h-1, w-1] - img_extend[h, w-1] - img_extend[h+1, w-1])
            mag = math.sqrt(p1*p1 + p2*p2)
    
            if mag < threshold:
                img_result[h-1, w-1] = 255 # else will be zero
        
    cv2.imwrite('b-prewitt.bmp', img_result)

    return img_result


def sobel_operator(img, threshold):
    print('Sobel\'s edge detection with threshold ' + str(threshold))

    height, width = img.shape[:2]
    img_result = np.zeros(img.shape[:2], dtype=np.uint8)

    # Extend the image
    # The height & width will be the original's plus 2
    img_extend = extend_image(img)

    for h in range(1, height + 1):
        for w in range(1, width + 1):
            p1 = (img_extend[h+1, w-1] + 2*img_extend[h+1, w] + img_extend[h+1, w+1] 
                 - img_extend[h-1, w-1] - 2*img_extend[h-1, w] - img_extend[h-1, w+1])
            p2 = (img_extend[h-1, w+1] + 2*img_extend[h, w+1] + img_extend[h+1, w+1]
                 - img_extend[h-1, w-1] - 2*img_extend[h, w-1] - img_extend[h+1, w-1])
            mag = math.sqrt(p1*p1 + p2*p2)
    
            if mag < threshold:
                img_result[h-1, w-1] = 255 # else will be zero
    
    cv2.imwrite('c-sobel.bmp', img_result)

    return img_result


def extend_image(img):
    height, width = img.shape[:2]
    img_extend = np.zeros((height + 2, width + 2), dtype=int)
    for h in range(height):
        for w in range(width):
            img_extend[h + 1, w + 1] = img[h, w]
            
    # 4 corner of the extended image
    img_extend[0, 0] = img_extend[1, 1]
    img_extend[0, width + 1] = img_extend[1, width]
    img_extend[height + 1, 0] = img_extend[height, 1]
    img_extend[height + 1, width + 1] = img_extend[height, width]

    # 4 edges of the extended image
    for c in range(1, width + 1):
        img_extend[0, c] = img_extend[1, c]
        img_extend[height + 1, c] = img_extend[height, c]
    for r in range(1, height + 1):
        img_extend[r, 0] = img_extend[r, 1]
        img_extend[r, width + 1] = img_extend[r, width]

    return img_extend

def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    # robert_operator(img, 30)
    # prewitt_operator(img, 24)
    sobel_operator(img, 38)
    

if __name__ == '__main__':
    main()