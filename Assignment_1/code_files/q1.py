import os
import sys
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import time
import cv2

# TODO: Looping code does not give correct results for reiszed images, fix that

image = cv2.imread(sys.argv[1], 0) # Read as grayscale
original_shape = image.shape
#print(image[100:200, 100:200])

cv2.imshow('image', image)
cv2.waitKey(0)
# cv2.imshow('crop', image[100:200, 100:200])
# cv2.waitKey(0)

def plot_bar_graph(labels, data, xlabel, ylabel, color, label):
    index = np.arange(len(labels))
    plt.bar(index, data, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(index, labels, fontsize=15, rotation=30)
    

def calculate_average_intensity_with_loops(image, input_shape=original_shape):
    start_time = time.time() # Function call started here

    
    total = 0
    width, height = input_shape #image.shape
    new_image = image if input_shape == original_shape else cv2.resize(image, input_shape, interpolation=cv2.INTER_AREA)
    
    total_pixels = width * height
    for i in range(width):
        for j in range(height):
            total += new_image[i][j]
    average_intensity = total / total_pixels 
    average_intensity = int(round(average_intensity))
    # print('Average intensity of the pixels is: {}'.format(average_intensity))
    # Thresholding on the basis of average intensity. [1 if intensity > average else 0]
    for i in range(width):
        for j in range(height):
            new_image[i][j] = 0 if new_image[i][j] < average_intensity else 1

    end_time = time.time() # Function's computations ended here
    plt.imshow(new_image, cmap='gray')
    plt.show()
    return (end_time - start_time) # time taken for function computation

def calculate_average_intensity_with_vectorization(image, input_shape=original_shape):
    start_time = time.time()

    # width, height = input_shape #image.shape
    if input_shape!=original_shape:
        image = cv2.resize(image, input_shape)

    average_intensity = np.mean(image, dtype='float16')
    average_intensity = int(round(average_intensity))
    image[image > average_intensity] = 1
    image[image <= average_intensity] = 0

    end_time = time.time()

    plt.imshow(image, cmap='gray')
    plt.show()
    return (end_time - start_time)

print("With Loops: ")
calculate_average_intensity_with_loops(image)
print("Without Loops: ")
calculate_average_intensity_with_vectorization(image)

shape_list = [[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)],
              ['16*16', '32*32', '64*64', '128*128', '256*256']]

looping_times = []
vectorization_times = []
for shape in shape_list[0]:
    lt = calculate_average_intensity_with_loops(image, shape)
    vt = calculate_average_intensity_with_vectorization(image, shape)
    looping_times.append(lt)
    vectorization_times.append(vt)

plot_bar_graph(shape_list[1], looping_times, 'Dimensions', 'Time (in seconds)', 'red', label='Looping Time')
plot_bar_graph(shape_list[1], vectorization_times, 'Dimensions', 'Time (in seconds)', 'blue', label='Vectorization Time')
plt.legend(loc='upper left')
plt.show()