import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from sys import argv

image = cv2.imread(argv[1])

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarize(image, threshold):
    binary_image = np.zeros(image.shape, dtype='uint8')
    binary_image[image >= threshold] = 1
    return binary_image

thresholds = [[0, 25], [50, 100], [150, 200], [250, 255]]

fig, ax = plt.subplots(nrows=4, ncols=2)

for i in range(4):
    ax[i, 0].imshow(binarize(image, thresholds[i][0]), cmap='gray', vmin=0, vmax=1)
    ax[i, 1].imshow(binarize(image, thresholds[i][1]), cmap='gray', vmin=0, vmax=1)

plt.show() 

mean_threshold = np.mean(image)

plt.imshow(binarize(image, mean_threshold), cmap='gray', vmin=0, vmax=1)
plt.title('For mean threshold value, the binary image is')
plt.show()