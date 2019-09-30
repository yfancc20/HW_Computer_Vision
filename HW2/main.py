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
    plt.savefig('b.png')
    # plt.show()


# (c)
def connected_components(img):
    print('(c) Finding connected components...')
    height, width = img.shape[:2]

    """ initialization """
    labels = np.zeros((height, width), dtype=np.int)
    labels_count = {}   # count the pixels of each label
    i = 0               # for new a label
    eq_class = []       # for equivalence class when detecting different labels

    """ top-down """
    for h in range(height):
        eq_class = [] # clean the array
        for w in range(width):
            # catch only white pixels
            if img[h, w][0] != 0: 
                # get neighbors of the current pixel 
                neighbors = get_neighbors(labels, h, w)

                # 4-connected, check the pixels up and left
                # When both up and left is empty, new a label then back to loop
                if not neighbors['up'] and not neighbors['left']:
                    i += 1
                    labels[h, w] = i
                    labels_count.update({labels[h, w] : 1})
                    continue

                # When at least neighbor exists, compare then propagate.
                if not neighbors['up']: 
                    labels[h, w] = neighbors['left']
                elif not neighbors['left']: 
                    labels[h, w] = neighbors['up']
                elif neighbors['up'] < neighbors['left']:
                    labels[h, w] = neighbors['up']
                    eq_class.append({
                        'old': neighbors['left'], 
                        'new': neighbors['up']
                    })
                else:
                    labels[h, w] = neighbors['left']

                labels_count[labels[h, w]] += 1
            # -- end of the loop of one row --

        # Deal with the in-line equivalence class 
        for eq in eq_class:
            for w in range(width):
                if labels[h, w] == eq['old']:
                    labels_count[eq['old']] -= 1
                    labels_count[eq['new']] += 1
                    labels[h, w] = eq['new']

    """ Bottom-up """
    change = True   # detect if a pixel change or not
    j = 0           # seed of choosing the direction when scanning rows
    while change:
        change = False
        j += 1
        for h in reversed(range(height)):
            for w in (reversed(range(width)) if j % 2 == 0 else range(width)):
                if labels[h, w]:
                    neighbors = get_neighbors(labels, h, w)
                    original_label = new_label = labels[h, w]

                    # Find min and propagate it 
                    for d in neighbors:
                        if neighbors[d] and labels[h, w] > neighbors[d]:
                            labels[h, w] = neighbors[d]
                            new_label = neighbors[d]
                            change = True

                    labels_count[original_label] -= 1
                    labels_count[new_label] += 1

    """ Draw the rectangle and cross """
    print("(c) Drawing the rectangles and crosses...")
    # Find the labels contain more than 500 pixels
    pass_labels = []
    for label in labels_count:
        if labels_count[label] >= 500 :
            pass_labels.append(label)

    # init of the rectangles' information
    rectangles = {}
    for label in pass_labels:
        rectangles.update({
            label: {
                'cent_x' : 0,
                'cent_y' : 0,
                'top': 512,
                'leftmost': 512,
                'bottom': -1,
                'rightmost': -1
            }
        })

    # Get the rectangle's info for each labels
    for label in pass_labels:
        total_x, total_y = 0, 0
        for h in range(height):
            for w in range(width):
                if labels[h, w] == label:
                    total_x += h
                    total_y += w
                    if h < rectangles[label]['top']:
                        rectangles[label]['top'] = h
                    if w < rectangles[label]['leftmost']:
                        rectangles[label]['leftmost'] = w 
                    if h > rectangles[label]['bottom']:
                        rectangles[label]['bottom'] = h 
                    if w > rectangles[label]['rightmost']:
                        rectangles[label]['rightmost'] = w 
        # end of scanning
        rectangles[label]['cent_x'] = (int)(total_x / labels_count[label])
        rectangles[label]['cent_y'] = (int)(total_y / labels_count[label])
        
    # Draw function
    cross_radius = 8
    for label, d in rectangles.items():
        # draw the rectangle
        cv2.rectangle(img, (d['leftmost'], d['top']), (d['rightmost'], d['bottom']), (255, 0, 0), 2)
        # find the middle point of each rectangle and extend it to a cross
        middle = (d['cent_x'], d['cent_y'])
        # draw the red cross
        for x in range(cross_radius):
            img[middle[0], middle[1] + x] = (0, 0, 255)
            img[middle[0], middle[1] - x] = (0, 0, 255)
            img[middle[0] + x, middle[1]] = (0, 0, 255)
            img[middle[0] - x, middle[1]] = (0, 0, 255)
    
    cv2.imwrite('c.bmp', img)

         
def get_neighbors(labels, h, w):
    return {
        'up': labels[h-1, w] if h - 1 >= 0 else 0,
        'left': labels[h, w-1] if w - 1 >= 0 else 0,
        'down': labels[h+1, w] if h + 1 < 512 else 0,
        'right': labels[h, w+1] if w + 1 < 512 else 0,
    }


def main():
    print('Reading the image...')
    img = cv2.imread('lena.bmp')
    binary_img = binarize_image(np.copy(img))
    histogram_image(np.copy(img))
    connected_components(binary_img)


if __name__ == '__main__':
    main()