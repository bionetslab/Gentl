#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:15:57 2024

@author: surya
"""
import matplotlib.pyplot as plt
import pydicom as dicom


# fig, ax=plt.subplots()

# ds=dicom.dcmread("./data/original/Al-Bladder Cancer/T0/CT-009/Lesion/1.2.840.113713.17.866.125132381139494410341856030984592697132.dcm")
# pixel_array_numpy=ds.pixel_array
# # plt.imshow(pixel_array_numpy, cmap='gray')



pixel_array_numpy_normal=plt.imread("./data/original/Al-Bladder Cancer/T1+/CT-169/Control/mask.png")
pixel_array_numpy=plt.imread("./data/original/Al-Bladder Cancer/T1+/CT-169/Lesion/CT-169.jpg")
mask=plt.imread("./data/original/Al-Bladder Cancer/T1+/CT-169/Lesion/mask.png")
# plt.imshow(mask)

# plt.show()

from matplotlib import cm
import numpy as np

try:
    from PIL import Image, ImageChops
except ImportError:
    import Image

background_normal =  Image.fromarray(np.uint8(pixel_array_numpy_normal*255))
# background_normal = background_normal.rotate(180)
background_cancer =  Image.fromarray(np.uint8(cm.gist_earth(pixel_array_numpy)*255))
overlay = Image.fromarray(np.uint8(mask*255))
# overlay = overlay.rotate(180)

new_img = Image.blend(background_cancer, overlay, 0.5)
new_img = Image.blend(new_img, background_normal, 0.5)
new_img.save("new_169.png","PNG")

plt.imshow(new_img)












dataset_path = "./data/data_as_jpg"
    output_path = "../data/processed_data"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # create a list to store rectangle coordinates
    data_list = []

    # Traverse the dataset folder
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):  # Check if the file is a .jpg file
                image_path = os.path.join(root, file)

                # Step 1: Read the image
                original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale





