from collections import Counter
from sys import argv
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

image = cv2.imread(argv[1], 0)

def generate_histogram(image):
    image_as_array = image.flatten()
    frequency_count = Counter(image_as_array)
    sorted_data = sorted(frequency_count.items())
    key_value_list = np.array([[k,v] for k,v in sorted_data]).T
    return key_value_list

def plot_histogram_v1(data):
    #f, axarr = plt.subplots(2, sharex=True)
    plt.plot(data[0], data[1], 'b')
    plt.xlim([0,255])
    plt.xlabel('Pixel intensities')
    plt.ylabel('Frequencies')
    plt.title('Histogram')

    plt.hist(image.flatten(), 256, range=(0, 256), color='r')
    plt.show()

hist = generate_histogram(image)

plot_histogram_v1(hist)