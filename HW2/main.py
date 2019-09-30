""" Homework 2 - Basic Image Manipulation
Yi-Fan Wu (R08921104)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

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
    height, width = img.shape[:2]

    # initialization
    labels = np.zeros((height, width), dtype=np.int)
    labels_count = {}
    i = 0 # increasing number from label
    # top-down
    for h in range(height):
        eq_class = []
        for w in range(width):
            if img[h, w][0] != 0: # catch the white pixel
                neighbors = get_neighbors(labels, h, w)
                if not neighbors['up'] and not neighbors['left']: # top-left
                    i += 1
                    labels[h, w] = i
                    color = list(np.random.choice(range(256), size=3))
                    img[h, w] = color
                    labels_count.update({labels[h, w] : 1})
                elif not neighbors['up']: # at least one neighbor exists
                    labels[h, w] = neighbors['left']
                    img[h, w] = img[h, w-1]
                elif not neighbors['left']: 
                    labels[h, w] = neighbors['up']
                    img[h, w] = img[h-1, w]
                elif neighbors['up'] < neighbors['left']:
                    labels[h, w] = neighbors['up']
                    eq_class.append((neighbors['left'], neighbors['up'], img[h-1, w]))
                    img[h, w] = img[h-1, w]
                    # img[h, w] = list(np.random.choice(range(256), size=3))
                else:
                    labels[h, w] = neighbors['left']
                    img[h, w] = img[h, w-1]
                labels_count[labels[h, w]] += 1
        
        if eq_class:
            for eq in eq_class:
                for w in range(width):
                    if labels[h, w] == eq[0]:
                        labels_count[labels[h, w]] -= 1
                        labels[h, w] = eq[1]
                        labels_count[eq[1]] += 1
                        img[h, w] = eq[2]

    change = True
    j = 0
    while change:
        change = False
        j += 1
        for h in reversed(range(height)):
            for w in (reversed(range(width)) if j % 2 == 0 else range(width)):
                if labels[h, w]:
                    neighbors = get_neighbors(labels, h, w)
                    original_label = new_label = labels[h, w]
                    direction = None
                    for d in neighbors:
                        if neighbors[d] and labels[h, w] > neighbors[d]:
                            labels[h, w] = neighbors[d]
                            new_label = neighbors[d]
                            direction = d
                            change = True
                    
                    if direction == 'right':
                        img[h, w] = img[h, w+1]
                    elif direction == 'down':
                        img[h, w] = img[h+1, w]
                    elif direction == 'left':
                        img[h, w] = img[h, w-1]

                    labels_count[original_label] -= 1
                    labels_count[new_label] += 1

    top_labels = []
    for key in labels_count:
        if labels_count[key] >= 500 :
            top_labels.append(key)

    rectangles = {}
    for key in top_labels:
        # []: top, left, down, right
        rectangles.update({key: [512, 512, -1, -1]})

    for key in top_labels:
        for h in range(height):
            for w in range(width):
                if labels[h, w] == key:
                    if h < rectangles[key][0]:
                        rectangles[key][0] = h
                    if w < rectangles[key][1]:
                        rectangles[key][1] = w
                    if h > rectangles[key][2]:
                        rectangles[key][2] = h
                    if w > rectangles[key][3]:
                        rectangles[key][3] = w
    
    print(rectangles)
    for key, value in rectangles.items():
        cv2.rectangle(img, (value[1], value[0]), (value[3], value[2]), (255, 0, 0), 2)
        middle = ((int)((value[2] + value[0]) / 2), (int)((value[3] + value[1]) / 2))
        # print(middle)
        for x in range(5):
            img[middle[0], middle[1] + x] = (0, 0, 255)
            img[middle[0], middle[1] - x] = (0, 0, 255)
            img[middle[0] + x, middle[1]] = (0, 0, 255)
            img[middle[0] - x, middle[1]] = (0, 0, 255)
    
    cv2.imwrite('ddd.bmp', img)

                
def get_neighbors(labels, h, w):
    return {
        'up': labels[h-1, w] if h - 1 >= 0 else 0,
        'left': labels[h, w-1] if w - 1 >= 0 else 0,
        'down': labels[h+1, w] if h + 1 < 512 else 0,
        'right': labels[h, w+1] if w + 1 < 512 else 0,
    }

def min_neighbor(labels, h, w):
    # 4-connected
    # up, right, down, left
    x = [-1, 0, 1, 0]
    y = [0, 1, 0, -1]
    min_label = labels[h, w]
    for i in range(4):
        if h + x[i] >= 0 and w + y[i] >= 0 \
           and labels[h+x[i], w+y[i]] != 0 \
           and labels[h+x[i], w+y[i]] < labels[h, w]:
           min_label = labels[h+x[i], w+y[i]]
    return min_label





def main():
    print('Reading the image...')
    # img = cv2.imread('lena.bmp')
    # binary_img = binarize_image(np.copy(img))
    # histogram_image(np.copy(img))
    img = cv2.imread('a.bmp')
    connected_components(img)


if __name__ == '__main__':
    main()