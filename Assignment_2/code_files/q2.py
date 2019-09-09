import cv2
import matplotlib.pyplot as plt        
import numpy as np

image = cv2.imread('../input_images/barbara.jpg', 0)

filters = {
    'sobel_h': 0.125*np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    'sobel_v': 0.125*np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),
    'prewitt_h': np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
    'prewitt_v': np.array([[1,1,1],[0,0,0],[-1,-1,-1]]),
    'roberts_h': np.array([[0,1],[-1,0]]),
    'roberts_v': np.array([[1,0],[0,-1]]),
    'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),
    'laplacian_d': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
}

def padding(image, filter_size):
    width, height = image.shape 
    padding_thickness = filter_size//2
    # vertically and horizontally stack 0 arrays
    hzeros = np.zeros(shape=(width, padding_thickness))
    image = np.hstack((hzeros, image, hzeros))
    vzeros = np.zeros(shape=(padding_thickness, height+2*padding_thickness))
    image = np.vstack((vzeros, image, vzeros))
    return image

def round_off(image):
    return np.around(image)

def apply_filter(image, _filter, filter_name):
    image_after_filtering = np.zeros(shape=image.shape, dtype=np.uint8)
    filter_size = len(_filter)
    initial_width, initial_height = image.shape
    image = padding(image, filter_size)
    width, height = image.shape
    for x in range(initial_height):
        for y in range(initial_width):
            correlation_value = np.dot(image[x:x+filter_size, y:y+filter_size].flatten(), _filter.flatten())
            image_after_filtering[x][y] = correlation_value
    image_after_filtering = round_off(image_after_filtering)
    plt.imshow(image_after_filtering, cmap='gray')
    plt.title(filter_name)
    plt.show()
    

apply_filter(image, filters['prewitt_h'], 'prewitt_h')  
apply_filter(image, filters['prewitt_v'], 'prewitt_v')
apply_filter(image, filters['laplacian'], 'laplacian')
apply_filter(image, filters['laplacian_d'], 'laplacian_d')