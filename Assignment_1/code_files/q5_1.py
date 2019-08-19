from sys import argv
import numpy as np 
import cv2
import matplotlib.pyplot as plt 

image = cv2.imread(argv[1], 0)
# Display the image
cv2.imshow('input image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def histogram_equalization_with_opencv(image):
    equalized_image = cv2.equalizeHist(image)
    f, axarr = plt.subplots(1, 2, sharey=True)
    axarr[0].imshow(image, cmap='gray')
    axarr[0].set_title('Original image')
    axarr[1].imshow(equalized_image, cmap='gray')
    axarr[1].set_title('Equalized image')
    plt.show()

def histogram_equalization_without_opencv(image):
    # create Histogram of image
    flat_image = image.flatten()
    histogram, _ = np.histogram(flat_image, 256, range=(0, 256))
    plt.hist(flat_image, 256, [0, 256], color='r')
    plt.xlim([0,256])
    plt.xlabel('Pixel Intensities')
    plt.ylabel('Frequency')
    plt.title('Original Histogram')
    plt.show()
    histogram = np.array(histogram) / (image.shape[0] * image.shape[1]) # produces normalized histogram
    # i.e. every element is divided by (M*N) here itself.

    # Now the cumulative frequency is calculated
    cumulative_sum = np.array([sum(histogram[:i+1]) for i in range(len(histogram))])
                                                             #                       k                    
    transformation_function = np.uint8(255 * cumulative_sum) # sk = T(rk) => (L-1)(Sigma nj) -> 255 * cumulative_sum 
                                                             #                      j=1
    transformed_image = np.zeros(shape=image.shape) # the tranformed image is stored in Y

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            transformed_image[i,j] = transformation_function[image[i, j]]

    H,_ = np.histogram(transformed_image.flatten(), 256, range=(0, 256)) # H is the equalized histogram
    plt.hist(transformed_image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0,256])
    plt.xlabel('Pixel Intensities')
    plt.ylabel('Frequency')
    plt.title('Equalized Histogram')
    plt.show()
    H = np.array(H) / (image.shape[0] * image.shape[1])
    f, axarr = plt.subplots(1, 2, sharey=True)
    axarr[0].imshow(image, cmap='gray')
    axarr[0].set_title('Original image')
    axarr[1].imshow(transformed_image, cmap='gray')
    axarr[1].set_title('Equalized image')
    plt.show()

histogram_equalization_with_opencv(image)

histogram_equalization_without_opencv(image)