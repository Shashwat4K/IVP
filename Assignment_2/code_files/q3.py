import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from q2 import padding
from sys import argv
# TODO: Image quality is not as expected

filters = {
    'laplacian' : np.array([[0,-1,0],[-1,4,1],[0,-1,0]]),
    'sobel_h' : 0.125*np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    'sobel_v' : 0.125*np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),
    'averaging': 0.04*np.ones(shape=(5,5))
}


def dot_product(a, b):
    length = len(a)
    s = 0
    for i in range(length):
        s += np.dot(a[i], b[i])
    return s     

def apply_filter(image, _filter, filter_name=None):
    image_after_filtering = np.zeros(shape=image.shape)
    filter_size = len(_filter)
    initial_height, initial_width = image.shape
    image = padding(image, _filter.shape)
    filter_height, filter_width = _filter.shape
    height, width = image.shape
    for x in range(height-filter_height):
        for y in range(width-filter_width):
            correlation_value = dot_product(image[x:x+filter_height, y:y+filter_width], _filter)
            image_after_filtering[x][y] = correlation_value
    return image_after_filtering 

def power_law(image, gamma):
    s = np.array(image**gamma)
    return s

if __name__ == '__main__':

    image = cv2.imread(argv[1], 0)
    height, width = image.shape
    
    # Step 1: APPLY LAPLACIAN FILTER
    print("Step 1: Applying Laplacian filter")
    laplacian_filtered_image = apply_filter(image, filters['laplacian'])
    # Step 2: SUBTRACT THE FILTERED IMAGE FROM ORIGINAL IMAGE
    print("Step 2: Image sharpening")
    sharpened_image = np.absolute(image - laplacian_filtered_image)
    # Step 3: APPLY SOBEL FILTERS (VERTICAL AND HORIZONTAL) AND ADD THEM
    print("Step 3: Applying Sobel filters")
    sobel_filtered_image = np.absolute(apply_filter(sharpened_image, filters['sobel_h']) + apply_filter(sharpened_image, filters['sobel_v']))
    # Step 4: APPLY AVERAGING FILTER TO SMOOTH THE IMAGE
    print("Step 4: Applying Averaging filter")
    smooth_image = apply_filter(sobel_filtered_image, filters['averaging'])
    # Step 5: DO PRODUCT OF LAPLACIAN FILTERED AND SMOOTHED IMAGE
    print("Step 5: Product")
    product_image = laplacian_filtered_image * smooth_image
    # Step 6: ADD THE PRODUCT IMAGE TO ORIGINAL IMAGE
    print("Step 6: Sum")
    sum_image = product_image + image
    # Step 7: APPLY POWER LAW TRANSFORM WITH GAMMA < 1, e.g. 0.5
    print('Step 7: Power law transform (0.28)')
    transformed_image = power_law(sum_image, 0.28)
    transformed_image = np.nan_to_num(transformed_image, 0.)
    figure = plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.subplot(1,2,2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title('Transformed image')
    plt.show()
    plt.imshow(laplacian_filtered_image, cmap='gray')
    plt.show()