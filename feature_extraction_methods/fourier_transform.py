import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from feature_extraction_methods import dummy_roi_extractor

bbox_region_image = dummy_roi_extractor.extract_cancer_roi()
print("Original Image Length",len(bbox_region_image.reshape(-1)))
print(bbox_region_image.reshape(-1),"\n")
DFT = cv.dft(np.float32(bbox_region_image), flags=cv.DFT_COMPLEX_OUTPUT)

# reposition the zero-frequency component to the spectrum's middle
fourier_shift = np.fft.fftshift(DFT)

# calculate the magnitude of the Fourier Transform - intensity of each frequency component
magnitude = 20 * np.log(cv.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))

# Scale the magnitude for display
magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1) # range between 0 and 255
print("Transform length", len(magnitude.reshape(-1)))
print(magnitude.reshape(-1),"\n")

# compute the center of the image
row, col = bbox_region_image.shape
center_row, center_col = row // 2, col // 2

# create a mask with a centered square of 1s
mask = np.ones((row, col, 2), np.uint8)
mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 0  # use high pass filter - only high
# frequency pass through

# put the mask and inverse DFT in place.
fft_shift = fourier_shift * mask
fft_ifft_shift = np.fft.ifftshift(fft_shift)
image_after_IFT = cv.idft(fft_ifft_shift)

# calculate the magnitude of the inverse DFT
image_after_IFT = cv.magnitude(image_after_IFT[:, :, 0], image_after_IFT[:, :, 1])
print("Feature Length", len(image_after_IFT.reshape(-1)))
print(image_after_IFT.reshape(-1))


"""Visualization code"""

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(bbox_region_image, cmap='gray')
axes[0].set_title(f"Cancer ROI")
axes[0].axis('off')

axes[1].imshow(magnitude, cmap='gray')
axes[1].set_title('Fourier Transform of an image')
axes[1].axis('off')

axes[2].imshow(image_after_IFT, cmap='gray')
axes[2].set_title('Inverse Fourier Transform of an image')
axes[2].axis('off')

plt.tight_layout()
plt.show()
