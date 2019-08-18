import sys
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import math
import time

# TODO: The Functions are incomplete and log_transformation function
#       has to be updated, because it's producing error (Unsupported datatype)     

spine_image = cv2.imread(sys.argv[1], 0)
#print(spine_image)
cv2.imshow('Original Spine Image', spine_image)
cv2.waitKey(0)

def log_transformation(image):
    '''
    transformed_image = np.zeros(image.shape, dtype='float16')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            transformed_image[i][j] = c * math.log(image[i,j] + 1.)
    '''
    c = np.ceil(255 / (np.log(1 + np.max(image))))
    log_image = c * (np.log(image + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    # print('MIN {} MAX {}'.format(transformed_image.min(), transformed_image.max()))        
    # print(log_image[500:600, 500:600], '\nMAX{} MIN{}'.format(log_image.max(), log_image.min()))
    figure = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')

    plt.subplot(1,2,2)
    plt.imshow(log_image, cmap='gray', vmin=log_image.min(), vmax=log_image.max())
    
    plt.show()
    
def power_transformation(image, c, gamma):
    spine_gamma = np.array(255 * (image/255)**gamma, dtype=np.uint8)
    
    figure = plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')

    plt.subplot(1,2,2)
    plt.imshow(spine_gamma, cmap='gray')

    plt.show()

log_transformation(spine_image)

power_transformation(spine_image, 1, 2.2)
power_transformation(spine_image, 1, 0.4)