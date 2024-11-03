import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.neighbors import KDTree
from math import dist

original_image = cv.imread('cancer2.jpg')
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
mask_image = cv.imread('mask2.png', cv.IMREAD_GRAYSCALE).astype(np.float32)
mask_image = (255.0 - mask_image) / 255.0  # reversing the mask
gray_image = gray_image.squeeze()
mask_image = mask_image.squeeze()

# print(np.max(mask_image))
# print(np.min(mask_image))
# plt.imshow(mask_image, cmap=plt.cm.gray)
# plt.show()

roi_width = 128
roi_height = 128
overlap = 0.25
max_rois_per_image = 12


def extract_cancer_roi():
    image = gray_image
    mask = mask_image

    bbox = compute_bounding_box(mask)

    # Extract pixel values within the bounding box where mask is 255
    rmin, cmin, rmax, cmax = bbox
    bbox_region_image = image[rmin:rmax, cmin:cmax]  # Image region within bbox

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Image")
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    axes[2].imshow(image, cmap='gray')
    axes[2].add_patch(
        plt.Rectangle(
            (bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
            fill=False, edgecolor='red', linewidth=1
            )
        )
    axes[2].set_title('Image with Bounding Box')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return bbox_region_image


def extract_non_cancer_rois():
    """
    Extract non-cancer ROIs from the image with specified overlap.

    Args:
    image (numpy.ndarray): Input image array.
    mask (numpy.ndarray): Binary mask where 1 indicates cancer regions.
    roi_width (int): Width of the ROI to extract.
    roi_height (int): Height of the ROI to extract.
    overlap (float): Overlap between ROIs, value between 0 and 1.
    max_rois (int, optional): Maximum number of ROIs to extract. If None, extract all possible ROIs.

    Returns:
    list: List of extracted ROI arrays.
    """
    h, w = gray_image.shape
    stride_y = int(roi_height * (1 - overlap))
    stride_x = int(roi_width * (1 - overlap))

    rois = []
    locations = []
    for y in range(0, h - roi_height + 1, stride_y):
        for x in range(0, w - roi_width + 1, stride_x):
            roi = gray_image[y:y + roi_height, x:x + roi_width]
            roi_mask = mask_image[y:y + roi_height, x:x + roi_width]
            # Check if the ROI is completely non-cancer
            # if np.sum(roi_mask) != 0:
            #     print(np.sum(roi_mask))
            #     print("X axis")
            #     print(x,x + roi_width)
            #     print("Y axis")
            #     print(y,y + roi_height)
            #     plt.imshow(roi, cmap=plt.cm.gray)
            #     plt.show()
            if np.sum(roi_mask) == 0:
                locations.append((y, x, y + roi_height, x + roi_width))
                rois.append(roi)

            if max_rois_per_image and len(rois) >= max_rois_per_image:
                neighbors = compute_neighbors(locations)
                print(neighbors)
                return rois, locations

    return rois, locations


import matplotlib.pyplot as plt
import numpy as np


def visualize_non_cancerous_region():
    # Assuming gray_image and mask_image are already defined
    image = gray_image  # Grayscale image
    mask = mask_image  # Mask image

    # Compute bounding boxes
    _, bbox = extract_non_cancer_rois()

    # Plot setup
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the grayscale image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Image")
    axes[0].axis('off')

    # Display the mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    # Display the image with all bounding boxes
    axes[2].imshow(image, cmap='gray')
    for region in bbox:
        rmin, cmin, rmax, cmax = region
        axes[2].add_patch(
            plt.Rectangle(
                (cmin, rmin), cmax - cmin, rmax - rmin,
                fill=False, edgecolor='red', linewidth=1
                )
            )
    axes[2].set_title('Image with Bounding Boxes')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def compute_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax


def compute_neighbors(locations):
    roi_coordinates = [(x, y) for (y, x, _, _) in locations]
    kdt_tree = KDTree(roi_coordinates, leaf_size=30)  # metric='euclidean'
    distances, indices = kdt_tree.query(roi_coordinates, k=4)
    neighbor_indices = indices[:, 1:]  # Remove first column (self-reference)
    neighbor_distances = distances[:, 1:]  # Remove first column (self-reference)

    # for each roi in roi_coordinates, store its neighbor and the distance as key value pair
    neighbors_dict = {
        i: [{int(idx): float(dist)} for idx, dist in zip(neighbor_indices[i], neighbor_distances[i])]
        for i in range(len(roi_coordinates))
        }

    return neighbors_dict


visualize_non_cancerous_region()
