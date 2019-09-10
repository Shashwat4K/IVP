import cv2
import matplotlib.pyplot as plt        
import numpy as np
from sys import argv

image = cv2.imread(argv[1], 0)

# TODO: Verify the results

filters = {
    'sobel_h': 0.125*np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    'sobel_v': 0.125*np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),
    'prewitt_h': np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
    'prewitt_v': np.array([[1,1,1],[0,0,0],[-1,-1,-1]]),
    'roberts_h': np.array([[0,1,0],[-1,0,0],[0,0,0]]),
    'roberts_v': np.array([[1,0,0],[0,-1,0],[0,0,0]]),
    'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
    'laplacian_d': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
}

def padding(image, filter_size):
    height, width = image.shape 
    padding_thickness = filter_size//2
    # vertically and horizontally stack 0 arrays
    hzeros = np.zeros(shape=(height, padding_thickness))
    image = np.hstack((hzeros, image, hzeros))
    vzeros = np.zeros(shape=(padding_thickness, width+2*padding_thickness))
    image = np.vstack((vzeros, image, vzeros))
    return image

def round_off(image):
    return np.around(image)

def apply_filter(image, _filter, filter_name):
    image_after_filtering = np.zeros(shape=image.shape)
    filter_size = len(_filter)
    initial_height, initial_width = image.shape
    image = padding(image, filter_size)
    height, width = image.shape
    for x in range(height-filter_size):
        for y in range(width-filter_size):
            correlation_value = np.dot(image[x:x+filter_size, y:y+filter_size].flatten(), _filter.flatten())
            image_after_filtering[x][y] = correlation_value
    image_after_filtering = round_off(image_after_filtering)
    plt.subplot(1,2,1)
    plt.imshow(image_after_filtering, cmap='gray')
    plt.title('Filter applied ' + filter_name)
    plt.subplot(1,2,2)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.show()
    
figure = plt.figure()
for (key, value) in filters.items():
    apply_filter(image, value, key)