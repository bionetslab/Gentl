import cv2 as cv
import matplotlib.pyplot as plt
import dummy_roi_extractor


# Call the function and get the bounding box region of the image
bbox_region_image = dummy_roi_extractor.extract_cancer_roi()

# Compute gradient magnitude and orientation using Sobel operators
grad_x = cv.Sobel(bbox_region_image, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(bbox_region_image, cv.CV_32F, 0, 1, ksize=3)
mag, angle = cv.cartToPolar(grad_x, grad_y, angleInDegrees=True)

"""Sift Descriptor Code"""
sift = cv.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(bbox_region_image, None)  # keypoints and corresponding descriptors
print(len(keypoints))

# Compute GLOH features for each keypoint
gloh_features = []
for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    scale = int(kp.size / 2)
    histogram = cv.calcHist([angle[y-scale:y+scale, x-scale:x+scale]], [0], None, [36], [0, 360])
    gloh_features.append(histogram)

# Concatenate the GLOH features into a single feature vector
gloh_features = cv.normalize(cv.hconcat(gloh_features), None)

print(gloh_features)