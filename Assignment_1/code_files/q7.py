from sys import argv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt 
import cv2

image = cv2.imread(argv[1], 0)

def k_dominant_intensities(image, k=1):
    histogram_data = np.array(sorted([(v, k) for (k, v) in Counter(image.flatten()).items()]))
    return np.array(histogram_data[-k:][::-1])
    
info = k_dominant_intensities(image, int(argv[2]))

# dominant_intensity = info[0][1]
dominant_intensity = info[len(info)][1]

k_top_intensities = np.array([j for (i,j) in info])

figure = plt.figure()

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.imshow([[dominant_intensity]], cmap='gray', vmin=0, vmax=255)

plt.subplot(1, 3, 3)
plt.imshow([k_top_intensities], cmap='gray', vmin=0, vmax=255)

plt.show()
