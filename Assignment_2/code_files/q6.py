import cv2
import numpy as np
import matplotlib.pyplot as plt
from q2 import padding
from q3 import dot_product, apply_filter

def apply_mean_filter(image):
    height, width = image.shape
    mean_filtered_image = np.zeros(shape=image.shape)
    image = padding(image, (3,3))
    plt.imshow(image, cmap='gray')
    plt.title('Padded_image')
    plt.show()
    print('Padded shape', image.shape)
    for x in range(height-3):
        for y in range(width-3):
            mean_filtered_image[x,y] = np.around(np.mean(image[x:x+3, y:y+3].flatten()))
    return mean_filtered_image, mean_filtered_image.shape

def apply_median_filter(image):
    height, width = image.shape
    median_filtered_image = np.zeros(shape=image.shape)
    image = padding(image, (3,3))
    print('Padded shape', image.shape)
    for x in range(height-3):
        for y in range(width-3):
            median_filtered_image[x,y] = np.median(image[x:x+3, y:y+3].flatten())
    return median_filtered_image

def get_mean_square_error(filtered_image, original_image, half_number=1):
    height, width = original_image.shape
    original_image = original_image[:, :width//2] if half_number == 1 else original_image[:, width//2:]
    return np.mean((original_image-filtered_image)**2)

if __name__ == '__main__':
    original_image = cv2.imread('../input_images/img1original.tif', 0)
    noisy_image = cv2.imread('../input_images/img1noisy.tif', 0)

    cv2.imshow('Original', original_image)
    cv2.imshow('Noisy', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    h, w = noisy_image.shape
    average_filter = (1/9)*np.ones(shape=(3,3))
    average_filtered_image_half_1 = cv2.filter2D(noisy_image[:,:w//2], -1, average_filter)
    median_filtered_image_half_1 = cv2.medianBlur(noisy_image[:,:w//2], 3)
    average_filtered_image_half_2 = cv2.filter2D(noisy_image[:, w//2:], -1, average_filter)
    median_filtered_image_half_2 = cv2.medianBlur(noisy_image[:, w//2:], 3)

    Mean_MSE_half_1 = get_mean_square_error(average_filtered_image_half_1, original_image, 1)
    Median_MSE_half_1 = get_mean_square_error(median_filtered_image_half_1, original_image, 1)
    Mean_MSE_half_2 = get_mean_square_error(average_filtered_image_half_2, original_image, 2)
    Median_MSE_half_2 = get_mean_square_error(median_filtered_image_half_2, original_image, 2)
    print('Mean MSE half 1 = {}, half 2 = {}\nMedian MSE half 1 = {}, half 2 = {}'.format(Mean_MSE_half_1, Mean_MSE_half_2, Median_MSE_half_1, Median_MSE_half_2))
    best_of_half_1 = 'mean' if Mean_MSE_half_1 < Median_MSE_half_1 else 'median'
    best_of_half_2 = 'mean' if Mean_MSE_half_2 < Median_MSE_half_2 else 'median'
    print('Best filter for half 1 = {} and half 2 ={}'.format(best_of_half_1, best_of_half_2))
    best_image = np.hstack(((average_filtered_image_half_1 if best_of_half_1=='mean' else median_filtered_image_half_1),\
                (average_filtered_image_half_2 if best_of_half_2=='mean' else median_filtered_image_half_2)))
    error_image = original_image - best_image
    print('MSE between filtered and original image = {}'.format(np.mean((best_image - original_image)**2)))
    f = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy image')
    plt.subplot(1,3,3)
    plt.imshow(best_image, cmap='gray')
    plt.title('Best image')
    plt.show()

    f = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1,3,2)
    plt.imshow(best_image, cmap='gray')
    plt.title('Best image')
    plt.subplot(1,3,3)
    plt.imshow(error_image, cmap='gray')
    plt.title('Error (Difference) image')
    plt.show()