import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from sys import argv

def add_gaussian_noise(image, mean=1, stddev=0.1):
    noise_component = np.random.normal(mean, stddev, image.shape)
    noise_component = noise_component.reshape(image.shape)
    return noise_component + image, noise_component

def add_salt_and_pepper_noise(image, amount=0.004, salt_p=0.5, pepper_p=0.5):
    noisy_image = np.copy(image)
    # Salt 
    salt_amount = int(np.ceil(amount* image.size * salt_p))
    noisy_image[tuple([np.random.randint(0, i-1, salt_amount) for i in image.shape])] = 255
    # Pepper
    pepper_amount = int(np.ceil(amount * image.size * pepper_p))
    noisy_image[tuple([np.random.randint(0, i-1, pepper_amount) for i in image.shape])] = 0
    return noisy_image

if __name__ == '__main__':
    
    image = cv2.imread(argv[1], 0)
    
    print('Gaussian noise')
    figure = plt.figure()
    mean = float(input("Enter the value of mean: "))
    stddev = float(input("Enter the Value of Standard-deviation: "))
    noisy_image, noise_mask = add_gaussian_noise(image, mean, stddev)
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.subplot(1,3,2)
    plt.imshow(noise_mask, cmap='gray')
    plt.title('Noise mask')
    plt.subplot(1,3,3)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Image with Gaussian noise with mean={} and stddev={}'.format(mean, stddev))
    plt.show()

    print('Salt and Pepper noise')
    figure = plt.figure()
    salt_p = float(input("Enter the value of salt probability: "))
    pepper_p = float(input("Enter the Value of pepper probabilty: "))
    amount = float(input('Enter amount: '))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.subplot(1,2,2)
    plt.imshow(add_salt_and_pepper_noise(image, amount, salt_p, pepper_p), cmap='gray')
    plt.title('Image with Salt and Pepper noise with probabilities {} and {} respectively'.format(salt_p, pepper_p))
    plt.show()