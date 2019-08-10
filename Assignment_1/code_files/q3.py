import sys
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import math
import time

# TODO: The Functions are incomplete and log_transformation function
#       has to be updated, because it's producing error (Unsupported datatype)     

spine_image = cv2.imread(sys.argv[1], 0)
print(spine_image[500:600, 500:600])
cv2.imshow('Original Spine Image', spine_image[500:600, 500:600])
cv2.waitKey(0)

def log_transformation(image, c):
    transformed_image = np.zeros(image.shape, dtype='float16')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            transformed_image[i][j] = c * math.log(image[i,j] + 1.)

    print('MIN {} MAX {}'.format(transformed_image.min(), transformed_image.max()))        
    cv2.imshow('transformed image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def power_transformation(image, c, gamma):
    pass

log_transformation(spine_image, 1)