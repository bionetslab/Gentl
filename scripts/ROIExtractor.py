import os
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree


def extract_non_cancer_rois(neighbor_parm, ct_folder, image, mask, out_boundary, roi_width, roi_height, overlap,
                            max_rois=None):
    """
    Extract healthy ROIs from the image with specified overlap.

    Arguments:
    neighbor_parm: Specifies the method for determining neighbors of (ROIs)
    ct_folder: Specifies the patient_id or filename
    image (numpy.ndarray): Input image array.
    mask (numpy.ndarray): Binary mask where 1 indicates cancer regions.
    out_boundary (numpy.ndarray): Image array where backgrounds are marked with pixel values set to 0.
    roi_width (int): Width of the ROI to extract.
    roi_height (int): Height of the ROI to extract.
    overlap (float): Overlap between ROIs, value between 0 and 1.
    max_rois (int, optional): Maximum number of ROIs to extract. If None, extract all possible ROIs.

    Returns:
    list: List of extracted ROI arrays.
    """
    query = "bladder_region"  # choose the bladder region
    x_, y_, w_, h_ = get_coordinates_from_csv(ct_folder, query)
    w_ = w_ + x_
    h_ = h_ + y_
    # h, w = image.shape  # (512,512)
    stride_y = int(roi_height * (1 - overlap))
    stride_x = int(roi_width * (1 - overlap))

    rois = []
    locations = []  # store coordinates of the roi
    for y in range(y_, h_ - roi_height + 1, stride_y):
        for x in range(x_, w_ - roi_width + 1, stride_x):
            roi = image[y:y + roi_height, x:x + roi_width]
            roi_mask = mask[y:y + roi_height, x:x + roi_width]
            background_masked_image = out_boundary[y:y + roi_height, x:x + roi_width]

            # Check if the ROI is completely non-cancer
            if np.sum(roi_mask) == 0 and np.sum(background_masked_image) == 0:
                coordinates = (y, x, y + roi_height, x + roi_width)  # (y1,x1,y2,x2)
                locations.append((y, x, y + roi_height, x + roi_width))  # store all the coordinates of rois per image
                rois.append((roi, coordinates))  # tuple with roi and coordinates

            if max_rois and len(rois) >= max_rois:
                if max_rois > 1:
                    neighbors = compute_neighbors(locations) if neighbor_parm == "knn" else distance_threshold(
                        locations
                        )
                    # Update each entry in `rois` to include its corresponding neighbors
                    rois = [
                        (roi, coordinates, neighbors[idx])
                        for idx, (roi, coordinates) in enumerate(rois)
                        ]
                else:
                    rois = [
                        (roi, coordinates, [])
                        for idx, (roi, coordinates) in enumerate(rois)
                        ]
                c_roi, c_coordinates = extract_cancer_roi(image, mask)
                locations.append(c_coordinates)
                c_neighbors = compute_neighbors(locations, True) if neighbor_parm == "knn" else distance_threshold(
                    locations, True
                    )
                # compute_delaunay_triangulation(locations)
                #visualize_and_store_non_cancerous_region(image, locations[:-1], ct_folder)
                return rois, c_roi, c_coordinates, c_neighbors[len(locations) - 1]
    if max_rois > 1:
        neighbors = compute_neighbors(locations) if neighbor_parm == "knn" else distance_threshold(locations)
        # Update each entry in `rois` to include its corresponding neighbors
        rois = [
            (roi, coordinates, neighbors[idx])
            for idx, (roi, coordinates) in enumerate(rois)
            ]
    else:
        rois = [
            (roi, coordinates, [])
            for idx, (roi, coordinates) in enumerate(rois)
            ]
    c_roi, c_coordinates = extract_cancer_roi(image, mask)
    locations.append(c_coordinates)
    c_neighbors = compute_neighbors(locations, True) if neighbor_parm == "knn" else distance_threshold(locations, True)
    #visualize_and_store_non_cancerous_region(image, locations[:-1], ct_folder)
    # compute_delaunay_triangulation(locations)
    return rois, c_roi, c_coordinates, c_neighbors[len(locations) - 1]


def extract_cancer_roi(image, mask):
    """
    Extract pixel values for the cancer roi within the bounding box where mask is 1

    Arguments:
    image (numpy.ndarray): Input image array.
    mask (numpy.ndarray): Binary mask where 1 indicates cancer regions.

    Returns:
        cancer_roi: Array of pixel values
        bbox: (y1,x1,y2,x2) coordinates of the bounding box
    """
    # Extract pixel values within the bounding box where mask is 1
    bbox = compute_bounding_box(mask)
    r_min, c_min, r_max, c_max = bbox  # (y1,x1,y2,x2)
    cancer_roi = image[r_min:r_max, c_min:c_max]  # Slice the cancer region within image
    # print(f"cancer{cancer_roi.shape}")
    return cancer_roi, bbox


def compute_bounding_box(mask):
    """
    Compute the bounding box around the cancer roi given a mask.

    Argument:
    mask (numpy.ndarray): Binary mask where 1 indicates cancer regions

    Returns:
    coordinates of the bounding box - diagonal coordinates
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax


def compute_neighbors(locations, cancer_roi=False):
    """
    Use KDTree to find the neighbors index and distance to each neighbor,
    no.of.neighbors controlled by param k.

    Arguments:
    locations (list): Coordinates(y1,x1,y2,x2) of each ROI
    cancer_roi: Boolean value - True for finding cancer roi neighbors else find neighbors of non cancer roi

    Returns:
    dictionary of neighbors with index, distance to and coordinates of each neighbor
    """
    no_of_neighbors = 3
    roi_coordinates = [(x, y) for (y, x, _, _) in locations]
    kdt_tree = KDTree(roi_coordinates, leaf_size=30)  # metric='euclidean'

    if cancer_roi:
        distances, indices = kdt_tree.query(
            [roi_coordinates[-1]], k=no_of_neighbors
            )  # k=n, finds the nearest n-1 neighbors
        neighbor_indices = indices[:, 1:]  # Remove first column (self-reference)
        neighbor_distances = distances[:, 1:]  # Remove first column (self-reference)
        neighbors_dict = {
            len(roi_coordinates) - 1: [{int(idx): {'distance': float(dist), 'coordinates': locations[int(idx)]}}
                                       for idx, dist in zip(neighbor_indices.ravel(), neighbor_distances.ravel())]
            }  # zip two lists and access the corresponding values
    else:
        distances, indices = kdt_tree.query(roi_coordinates, k=no_of_neighbors)  # k=n, finds the nearest n-1 neighbors
        neighbor_indices = indices[:, 1:]  # Remove first column (self-reference)
        neighbor_distances = distances[:, 1:]  # Remove first column (self-reference)
        # for each roi in roi_coordinates, store its neighbor and the distance as key value pair
        neighbors_dict = {
            i: [{int(idx): {'distance': float(dist), 'coordinates': locations[int(idx)]}}
                for idx, dist in zip(neighbor_indices[i], neighbor_distances[i])]
            for i in range(len(roi_coordinates))
            }  # zip two lists and access the corresponding values
    return neighbors_dict


# {10: [{7: {'distance': 205.73040611440985, 'coordinates': (96, 192, 224, 320)}}, {8: {'distance': 209.88806540630173, 'coordinates': (96, 288, 224, 416)}}, {6: {'distance': 242.9588442514493, 'coordinates': (96, 96, 224, 224)}}]}
# 'neighbors': [{1: {'distance': 96.0, 'coordinates': (0, 96, 128, 224)}}, {2: {'distance': 192.0, 'coordinates': (0, 192, 128, 320)}}, {3: {'distance': 288.0, 'coordinates': (0, 288, 128, 416)}}]


def get_coordinates_from_csv(ct_folder, query):
    """
    From the csv file return x,y,width,height of bounding rectangular area for the bladder region within the entire CT scan

    Arguments:
    ct_folder: patient_id or filename
    query (string): bladder region for roi selection

    Returns:
    cod (tuple): x,y,width,height of rectangular area
    """
    full_data = pd.read_csv("../data/processed_data.csv", index_col=0)
    cod = full_data.loc[ct_folder, query]
    cod = literal_eval(cod)
    return cod


def distance_threshold(locations, cancer_roi=False):
    """
    Use distance threshold to find the neighbors index and distance to each neighbor,
    threshold is set using threshold_dist parameter.

    Arguments:
    locations (list): Coordinates(y1,x1,y2,x2) of each ROI
    cancer_roi (Boolean value): True for finding cancer roi neighbors else find neighbors of non cancer roi

    Returns:
    dictionary of neighbors with index, distance to and coordinates of each neighbor
    """
    threshold_dist = 10
    roi_coordinates = [(x, y) for (y, x, _, _) in locations]
    kdt_tree = KDTree(roi_coordinates, leaf_size=30)  # metric='euclidean'

    if cancer_roi:
        indices, distances = kdt_tree.query_radius(
            [roi_coordinates[-1]], r=threshold_dist, return_distance=True
            )  # return neighbors within the distance
        # neighbor_indices = indices[:, 1:]  # Remove first column (self-reference)
        # neighbor_distances = distances[:, 1:]  # Remove first column (self-reference)
        neighbors_dict = {
            len(roi_coordinates) - 1: [{int(idx): {'distance': float(dist), 'coordinates': locations[int(idx)]}}
                                       for idx, dist in zip(indices[0], distances[0]) if
                                       idx != len(roi_coordinates) - 1]
            }  # zip two lists and access the corresponding values
    else:
        indices, distances = kdt_tree.query_radius(roi_coordinates, r=threshold_dist, return_distance=True)
        # neighbor_indices = indices[:, 1:]  # Remove first column (self-reference)
        # neighbor_distances = distances[:, 1:]  # Remove first column (self-reference)
        # for each roi in roi_coordinates, store its neighbor and the distance as key value pair
        neighbors_dict = {
            i: [{int(idx): {'distance': float(dist), 'coordinates': locations[int(idx)]}}
                for idx, dist in zip(indices[i], distances[i]) if idx != i]
            for i in range(len(roi_coordinates))
            }  # zip two lists and access the corresponding values
    return neighbors_dict


# neighbors_dict = {
#     i: [{int(idx): {'distance': float(dist), 'coordinates': points[int(idx)]}}
#         for idx, dist in zip(all_nn_indices[i], distances[i]) if idx != i ] for i in range(len(points))
#     }


def visualize_and_store_non_cancerous_region(image, bbox, ct_folder):
    """
    Visualizes and store healthy ROIs with bounding boxes on a grayscale image and saves it to a file.

    Arguments:
    - image: 2D array-like grayscale image.
    - bbox: List of bounding boxes, where each box is a tuple (rmin, cmin, rmax, cmax).
    - ct_folder: Folder containing CT scans (for logging or processing purposes).
    """
    print(ct_folder)
    print(f"Bbox length {len(set(bbox))}")

    output_folder = "../data/with_40_bounding_boxes_within_bladder_region/"
    filename = f"{ct_folder}.jpg"
    output_path = os.path.join(output_folder, filename)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    # Add bounding boxes
    for region in bbox:
        rmin, cmin, rmax, cmax = region
        ax.add_patch(
            plt.Rectangle(
                (cmin, rmin), cmax - cmin, rmax - rmin,
                fill=False, edgecolor='red', linewidth=1
                )
            )

    # Remove axis labels, ticks, and spines
    ax.axis('off')

    # Save the plot to a file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    # Show the plot on screen
    # plt.show()

    # close the figure
    plt.close(fig)

