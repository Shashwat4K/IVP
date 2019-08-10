import cv2
import numpy as np 
import matplotlib.pyplot as plt 


image1 = cv2.imread('../input_images/img61.jpg')
image2 = cv2.imread('../input_images/Sample_Image.jpg')

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

cv2.imshow('First Image', image1)
cv2.waitKey(0)
cv2.imshow('Second Image', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

image2[image2 >= 240] = 0
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_CUBIC)

image1[image2 != 0] = 0
result = image1 + image2

plt.imshow(result)
plt.title('Result')
plt.show()