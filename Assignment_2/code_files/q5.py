import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os 

def padding(image):
    hzeros = np.zeros(shape=(image.shape[0], 1))
    image = np.hstack((hzeros, image, hzeros))
    vzeros = np.zeros(shape=(1, image.shape[1]))
    image = np.vstack((vzeros, image, vzeros))
    return image

def remove_salt_or_pepper_noise(image, is_salt=True):
    print('Applying {} filter'.format('\'Min\'' if is_salt==True else '\'Max\''))
    h, w = image.shape
    noise_free_image = np.zeros(shape=image.shape) 
    image = padding(image)
    for x in range(h-3):
        for y in range(w-3):
            noise_free_image[x,y] = np.amin(image[x:x+3, y:y+3]) if is_salt==True else np.amax(image[x:x+3, y:y+3])
    return noise_free_image

def remove_salt_and_pepper_noise(image):
    print('Applying \'Median\' filter')
    h, w = image.shape
    noise_free_image = np.zeros(shape=image.shape) 
    image = padding(image)
    for x in range(h-3):
        for y in range(w-3):
            noise_free_image[x,y] = np.median(image[x:x+3, y:y+3])
    return noise_free_image

def plot_results(before, after, image_name):
    f = plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(before, cmap='gray')
    plt.title('Original ' + image_name)
    plt.subplot(1,2,2)
    plt.imshow(after, cmap='gray')
    plt.title('Transformed ' + image_name)
    plt.show()
    plt.close(f)

ABSOLUTE_PATH = '/home/shashwat/Documents/IVP/Assignment_2/input_images'

if __name__ == '__main__':
    
    images = [cv2.imread(os.path.join(ABSOLUTE_PATH, i+'.jpg'), 0) for i in list('abcdefg')]
    
    # b.jpg has salt noise
    b_noise_free = remove_salt_or_pepper_noise(images[1])
    plot_results(images[1], b_noise_free, 'b.jpg')
    # c.jpg has pepper noise
    c_noise_free = remove_salt_or_pepper_noise(images[2], is_salt=False)
    plot_results(images[2], c_noise_free, 'c.jpg')
    # d.jpg has salt and pepper noise
    d_noise_free = remove_salt_and_pepper_noise(images[3])
    plot_results(images[3], d_noise_free, 'd.jpg')
    # f.jpg has pepper noise 
    f_noise_free = remove_salt_or_pepper_noise(images[5], is_salt=False)
    plot_results(images[5], f_noise_free, 'f.jpg')
    # g.jpg has salt noise
    g_noise_free = remove_salt_or_pepper_noise(images[6])
    plot_results(images[6], g_noise_free, 'g.jpg')