import cv2
import numpy as np 
import matplotlib.pyplot as plt
from sys import argv 
from q2 import padding
from q3 import dot_product

def template_matching(image, template):
    template_height, template_width = template.shape
    template_matching_map = np.zeros(shape=image.shape)
    padded_image = padding(image, template.shape)
    template_height, template_width = template.shape
    height, width = padded_image.shape
    for x in range(height-template_height):
        for y in range(width-template_width):
            t = dot_product(padded_image[x:x+template_height, y:y+template_width], template)
            template_matching_map[x,y] = t
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.subplot(1,3,2)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.subplot(1,3,3)
    plt.imshow(template_matching_map, cmap='gray')
    plt.title('Template matching map')
    plt.show()

if __name__ == '__main__':
    image = cv2.imread(argv[1], 0)
    template = cv2.imread(argv[2], 0)

    figure = plt.figure()

    template_matching(image, template)    