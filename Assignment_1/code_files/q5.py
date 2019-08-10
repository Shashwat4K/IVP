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
    
    # Build a Cumulative Distibution function
    cumulative_distribution = histogram.cumsum()
    normalized_cumulative_distribution = (cumulative_distribution * histogram.max())/ cumulative_distribution.max()
    
    # Plot the original histogram
    plt.plot(normalized_cumulative_distribution, color='b')
    plt.hist(flat_image, 256, [0, 256], color='r')
    plt.xlim([0,256])
    plt.xlabel('Pixel Intensities')
    plt.ylabel('Frequency')
    plt.title('Original Image histogram and cumulative ditribution function')
    
    plt.show()

    # Equalization operation
    '''
    equalized_image = np.zeros(shape=image.shape, dtype='uint8')
    width, height = image.shape
    for i in range(width):
        for j in range(height):
            equalized_image[i, j] = normalized_cumulative_distribution[image[i,j]]
    '''
    cumulative_distribution_masked = np.ma.masked_equal(cumulative_distribution, 0) # puts a mask wherever there is 0
    cumulative_distribution_masked = (cumulative_distribution_masked - cumulative_distribution_masked.min())*255/(cumulative_distribution_masked.max()-cumulative_distribution_masked.min())
    cumulative_distribution = np.ma.filled(cumulative_distribution_masked,0).astype('uint8')
    equalized_image = cumulative_distribution[image]

    # Plotting new histogram amd new cumulative distribution
    new_histogram, _ = np.histogram(equalized_image.flatten(), 256, range=(0, 256))
    plt.plot(cumulative_distribution, color='b')
    plt.hist(equalized_image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.xlabel('Pixel intensities')
    plt.ylabel('Frequencies')
    plt.title('Equalized image histogram')
    plt.show()  

    # See the difference
    f, axarr = plt.subplots(1, 2, sharey=True)
    axarr[0].imshow(image, cmap='gray')
    axarr[1].imshow(equalized_image, cmap='gray')
    axarr[0].set_title('Original image')
    axarr[1].set_title('See the difference')
    plt.show()

histogram_equalization_without_opencv(image)     
histogram_equalization_with_opencv(image) 