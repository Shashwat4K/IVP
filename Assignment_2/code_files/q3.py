import cv2
import numpy as np 
import matplotlib.pyplot as plt 

# TODO: Image quality is not as expected

image = cv2.imread('../input_images/skeleton_orig.tif', 0)
height, width = image.shape

filters = {
    'Laplacian' : np.array([[0,-1,0],[-1,4,1],[0,-1,0]]),
    'Sobel_h' : 0.125*np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    'Sobel_v' : 0.125*np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),
    'Averaging': 0.04*np.ones(shape=(5,5))
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

def apply_filter(image, _filter, filter_name=None):
    image_after_filtering = np.zeros(shape=image.shape)
    filter_size = len(_filter)
    initial_height, initial_width = image.shape
    image = padding(image, filter_size)
    height, width = image.shape
    for x in range(height-filter_size):
        for y in range(width-filter_size):
            correlation_value = np.dot(image[x:x+filter_size, y:y+filter_size].flatten(), _filter.flatten())
            image_after_filtering[x][y] = correlation_value
    return image_after_filtering 

def power_law(image, gamma):
    s = np.array(255*(image/255)**gamma)
    return s

# Step 1: APPLY LAPLACIAN FILTER
print("Step 1: Applying Laplacian filter")
laplacian_filtered_image = apply_filter(image, filters['Laplacian'])
# Step 2: SUBTRACT THE FILTERED IMAGE FROM ORIGINAL IMAGE
print("Step 2: Image sharpening")
sharpened_image = np.absolute(image - laplacian_filtered_image)
# Step 3: APPLY SOBEL FILTERS (VERTICAL AND HORIZONTAL) AND ADD THEM
print("Step 3: Applying Sobel filters")
sobel_filtered_image = np.absolute(apply_filter(sharpened_image, filters['Sobel_h']) + apply_filter(image, filters['Sobel_v']))
# Step 4: APPLY AVERAGING FILTER TO SMOOTH THE IMAGE
print("Step 4: Applying Averaging filter")
smooth_image = apply_filter(sobel_filtered_image, filters['Averaging'])
# Step 5: DO PRODUCT OF LAPLACIAN FILTERED AND SMOOTHED IMAGE
print("Step 5: Product")
product_image = laplacian_filtered_image * smooth_image
# Step 6: ADD THE PRODUCT IMAGE TO ORIGINAL IMAGE
print("Step 6: Sum")
sum_image = product_image + image 
'''
# Step 7: APPLY POWER LAW TRANSFORM WITH GAMMA < 1, e.g. 0.5
print('Step 7: Power law transform (0.5)')
transformed_image = power_law(sum_image, 0.5)
'''

figure = plt.figure()

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.subplot(1,2,2)
plt.imshow(transformed_image, cmap='gray')
plt.title('Transformed image')
plt.show()