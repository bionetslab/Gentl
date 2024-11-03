import cv2 as cv
import matplotlib.pyplot as plt
import dummy_roi_extractor

# Call the function and get the bounding box region of the image
bbox_region_image = dummy_roi_extractor.extract_cancer_roi()

"""Sift Descriptor Code"""
sift = cv.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(bbox_region_image, None)  # keypoints and corresponding descriptors
print(len(keypoints))
"""Visualization Code"""
# print(keypoints)
# print(descriptors)

img_with_keypoints = cv.drawKeypoints(bbox_region_image, keypoints, None)  # pass input image and keypoints

# Display the original image and key points

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].imshow(bbox_region_image, cmap='gray')
axes[0].set_title(f"Bounding Box Image")
axes[0].axis('off')

axes[1].imshow(img_with_keypoints, cmap='gray')
axes[1].set_title('Image with sift extracted key points')
axes[1].axis('off')

plt.tight_layout()
plt.show()
