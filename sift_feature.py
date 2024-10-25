import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
original_image = cv.imread('cancer.jpg')
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

"""Sift Descriptor Code"""
sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray_image, None)  # keypoints and corresponding descriptors


"""Visualization Code"""
# print(len(kp))
# print(des.shape[0]*des.shape[1])
img = cv.drawKeypoints(gray_image, kp, original_image)
# cv.imwrite('sift_keypoints.jpg',img)

plt.figure(figsize=(10, 2))

plt.subplot(121)  # Use 121 for the first subplot (1 row, 2 columns, 1st subplot)
plt.imshow(gray_image, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Original Image')

plt.subplot(122)  # Use 122 for the first subplot (1 row, 2 columns, 2nd subplot)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Image with keypoints')

plt.show()