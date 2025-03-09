import os
from torch.utils.data import Dataset
from scripts.ROIExtractor import extract_non_cancer_rois
import pydicom
from PIL import Image
import numpy as np
import torch


class BladderCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for main_folder in os.listdir(root_dir):
            main_path = os.path.join(root_dir, main_folder)
            # print("Main Path: ", main_path)
            if not os.path.isdir(main_path) or main_folder == 'Redo':
                continue  # continue if redo folder or if its not a directory
            else:
                self._process_folder(main_path, main_folder)

    def _process_folder(self, folder_path, time_point):
        for ct_folder in os.listdir(folder_path):
            ct_path = os.path.join(folder_path, ct_folder)
            if not os.path.isdir(ct_path):
                continue

            for case_type in ['Lesion']:
                case_path = os.path.join(ct_path, case_type)
                if not os.path.isdir(case_path):
                    continue

                dcm_file = None
                mask_file = None
                # coords_file = None

                for file in os.listdir(case_path):
                    if file.endswith('.dcm'):
                        dcm_file = os.path.join(case_path, file)
                    elif file.endswith('.png') and file == "mask.png":
                        mask_file = os.path.join(case_path, file)
                    elif file.endswith('.png') and file == "bladder_mask.png":  # bladder region
                        bladder_mask = os.path.join(case_path, file)
                    # elif file == 'coords.txt':
                    #     coords_file = os.path.join(case_path, file)

                if dcm_file and mask_file:  # and coords_file:
                    self.samples.append(
                        {
                            'dcm': dcm_file,
                            'mask': mask_file,
                            # 'coords': coords_file,
                            "background_masked_image": bladder_mask,  # bladder region with pixel 0
                            'time_point': time_point,
                            'ct_folder': ct_folder,
                            'case_type': case_type
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        dcm = pydicom.dcmread(sample['dcm'])
        image = dcm.pixel_array.astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min())

        mask = np.array(Image.open(sample['mask'])).astype(np.float32)
        mask = mask / 255.0

        background_masked_image = np.array(Image.open(sample['background_masked_image']))
        background_masked_image = background_masked_image / 255.0
        # with open(sample['coords'], 'r') as f:
        #     coords = f.read().strip()

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        background_masked_image = torch.from_numpy(background_masked_image).unsqueeze(0)

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {
            'image': image,
            'mask': mask,
            'background_masked_image': background_masked_image,
            # 'coords': coords,
            'time_point': sample['time_point'],
            'ct_folder': sample['ct_folder'],
            'case_type': sample['case_type']
            }


class BladderCancerROIDataset(Dataset):
    def __init__(self, base_dataset, roi_width, roi_height, overlap, max_rois_per_image=None):
        self.base_dataset = base_dataset
        self.roi_width = roi_width
        self.roi_height = roi_height
        self.overlap = overlap
        self.max_rois_per_image = max_rois_per_image

        self.roi_samples, self.cancer_samples = self._extract_all_rois()

    def _extract_all_rois(self):
        roi_samples = []
        cancer_samples = []
        neighbour_parm = "knn"  # or dist_threshold or knn
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]  # calling getitem() from BladderCancerDataset class
            image = sample['image'].squeeze().numpy()
            mask = sample['mask'].squeeze().numpy()
            background_masked_image = sample['background_masked_image'].squeeze().numpy()
            ct_folder = sample['ct_folder']

            rois, cancer_roi, cancer_coordinates, cancer_neighbors = extract_non_cancer_rois(
                neighbour_parm, ct_folder, image, mask, background_masked_image, self.roi_width, self.roi_height,
                self.overlap, self.max_rois_per_image
                )

            # Organize cancer samples by folder_name for quick lookup as a dictionary
            # eg: {'CT-009':cancer_roi,'CT-010':cancer_roi}
            cancer_samples.append(
                {
                    "index": f"C-{sample['ct_folder']}",
                    "image": torch.from_numpy(cancer_roi).float().unsqueeze(0),
                    "coordinates": tuple(int(cord) for cord in cancer_coordinates),
                    "neighbors": cancer_neighbors,
                    'time_point': sample['time_point'],  # cancer stage
                    'ct_folder': sample['ct_folder'],  # folder name
                    'case_type': sample['case_type']  # lesion
                    }

                )

            for roi_idx, (roi, coordinates, neighbors) in enumerate(
                    rois
                    ):  # store index,roi,coordinates and neighbors of each roi
                roi_samples.append(
                    {
                        'index': f"{roi_idx}-{sample['ct_folder']}",  # combine index and folder eg: (0-CT-009)
                        'roi': roi,
                        'coordinates': coordinates,  # coordinates of each roi
                        'neighbors': neighbors,  # neighbors of each roi {neighbor:(distance,coordinates)}
                        'time_point': sample['time_point'],  # cancer type eg :T0,T1+
                        'ct_folder': sample['ct_folder'],  # folder name eg: CT-009
                        'case_type': sample['case_type']  # Lesion
                        }
                    )

        return roi_samples, cancer_samples

    def __len__(self):
        return len(self.roi_samples)

    def __getitem__(self, idx):
        sample = self.roi_samples[idx]

        # Convert ROI to tensor and add channel dimension
        roi_tensor = torch.from_numpy(sample['roi']).float().unsqueeze(0)

        return {
            'index': sample['index'],  # non cancer roi index
            'image': roi_tensor,  # non cancer roi
            'coordinates': sample['coordinates'],  # coordinates of each roi
            'neighbors': sample['neighbors'],  # neighbors of each roi
            'time_point': sample['time_point'],  # cancer stage
            'ct_folder': sample['ct_folder'],  # folder name
            'case_type': "Control"  # control for healthy
            }

    def get_cancer_samples(self):
        """Returns list of dictionary of all cancer roi samples"""
        return self.cancer_samples

# 10 - 10,10,0.05
# 20 - 10,10,0.25
# 30 - 8,8,0.25
# 40 - 8,8,0.30
# 50 - 8,8,0.40
# """
