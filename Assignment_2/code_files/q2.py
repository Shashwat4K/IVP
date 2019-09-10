import cv2
import matplotlib.pyplot as plt        
import numpy as np
from sys import argv

# TODO: Verify the results

filters = {
    'sobel_h': 0.125*np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    'sobel_v': 0.125*np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),
    'prewitt_h': np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
    'prewitt_v': np.array([[1,1,1],[0,0,0],[-1,-1,-1]]),
    'roberts_h': np.array([[0,1,0],[-1,0,0],[0,0,0]]),
    'roberts_v': np.array([[1,0,0],[0,-1,0],[0,0,0]]),
    'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
    'laplacian_d': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
    'averaging': 0.04*np.ones(shape=(5,5))
}

def padding(image, filter_shape):
    height, width = image.shape 
    filter_height, filter_width = filter_shape
    vertical_padding_thickness = filter_height//2
    horizontal_padding_thickness = filter_width//2
    # vertically and horizontally stack 0 arrays
    hzeros = np.zeros(shape=(height, horizontal_padding_thickness))
    image = np.hstack((hzeros, image, hzeros))
    vzeros = np.zeros(shape=(vertical_padding_thickness, width+2*horizontal_padding_thickness))
    image = np.vstack((vzeros, image, vzeros))
    return image

def round_off(image):
    return np.around(image)

def apply_filter(image, _filter, filter_name):
    image_after_filtering = np.zeros(shape=image.shape)
    filter_size = len(_filter)
    filter_height, filter_width = _filter.shape 
    initial_height, initial_width = image.shape
    image = padding(image, _filter.shape)
    height, width = image.shape
    for x in range(height-filter_height):
        for y in range(width-filter_width):
            correlation_value = np.dot(image[x:x+filter_height, y:y+filter_width].flatten(), _filter.flatten())
            image_after_filtering[x][y] = correlation_value
    image_after_filtering = round_off(image_after_filtering)
    # image_after_filtering = np.absolute(image_after_filtering)
    plt.subplot(1,2,1)
    plt.imshow(image_after_filtering, cmap='gray')
    plt.title('Filter applied ' + filter_name)
    plt.subplot(1,2,2)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.show()

if __name__ == '__main__':  
    image = cv2.imread(argv[1], 0)  
    figure = plt.figure()
    for (key, value) in filters.items():
        apply_filter(image, value, key)