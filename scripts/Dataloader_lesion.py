import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scripts.ROIExtractor import extract_non_cancer_rois, BladderCancerROIVisualizer, extract_cancer_roi
import pydicom
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx


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


class BladderCancerVisualizer:
    @staticmethod
    def visualize_sample(sample):
        image = sample['image'].squeeze().numpy()
        mask = sample['mask'].squeeze().numpy()

        bbox = BladderCancerVisualizer.compute_bounding_box(mask)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f"DICOM Image\n{sample['time_point']} - {sample['ct_folder']} - {sample['case_type']}")
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

    @staticmethod
    def visualize_batch(batch, num_samples=4):
        batch_size = batch['image'].shape[0]
        num_samples = min(num_samples, batch_size)

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        for i in range(num_samples):
            image = batch['image'][i].squeeze().numpy()
            mask = batch['mask'][i].squeeze().numpy()
            bbox = BladderCancerVisualizer.compute_bounding_box(mask)

            axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title(
                f"DICOM Image\n{batch['time_point'][i]} - {batch['ct_folder'][i]} - {batch['case_type'][i]}"
                )
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title('Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(image, cmap='gray')
            axes[i, 2].add_patch(
                plt.Rectangle(
                    (bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                    fill=False, edgecolor='red', linewidth=1
                    )
                )
            axes[i, 2].set_title('Image with Bounding Box')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_bounding_box(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, cmin, rmax, cmax


def visualize_dataset(dataset, num_samples=4):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image'].squeeze().numpy()
        mask = sample['mask'].squeeze().numpy()
        bbox = BladderCancerVisualizer.compute_bounding_box(mask)

        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title(f"DICOM Image\n{sample['time_point']} - {sample['ct_folder']} - {sample['case_type']}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(image, cmap='gray')
        axes[i, 2].add_patch(
            plt.Rectangle(
                (bbox[1], bbox[0]), bbox[3] - bbox[1], bbox[2] - bbox[0],
                fill=False, edgecolor='red', linewidth=2
                )
            )
        axes[i, 2].set_title('Image with Bounding Box')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def draw_network_graph(graph_name,patient_label):
    """
    To draw network graph

    Arguments: graph_name - corresponding to each patient
    patient_label - corresponding to each patient
    """
    plt.figure(figsize=(10, 8))  # Adjust the figure size
    pos = nx.shell_layout(graph_name)  # Apply layout

    # Extract node attributes
    case_types = nx.get_node_attributes(graph_name, "case_type")

    # Map case types to colors
    color_map = {
        "Lesion": "red",  # Lesion nodes in red
        "Control": "green"  # Non-lesion nodes in green
    }

    node_colors = [color_map.get(case_types[node]) for node in graph_name.nodes]

    # Draw nodes, edges, and labels
    nx.draw_networkx_nodes(graph_name, pos, node_color=node_colors, node_size=1000)
    nx.draw_networkx_edges(graph_name, pos, edge_color='gray')

    # Add index as node values
    nx.draw_networkx_labels(graph_name, pos, font_size=12, font_color='black')

    # Extract edge labels and format them to two decimal places
    edge_labels = nx.get_edge_attributes(graph_name, "spatial_distance")
    formatted_edge_labels = {edge: f"{dist:.2f}" for edge, dist in edge_labels.items()}  # Format to 2 decimals

    # Draw edge labels with formatted values
    nx.draw_networkx_edge_labels(graph_name, pos, edge_labels=formatted_edge_labels, font_color='red')

    # Display the graph
    patient_id = patient_label.split(sep="-")[1]
    plt.title(f"Spatial Network patient#: {patient_id}")
    plt.axis('off')
    plt.show()


# base_dataset = BladderCancerDataset(
#     root_dir='../data/original/Al-Bladder Cancer'
#     )
# roi_per_image =40
# roi_dataset = BladderCancerROIDataset(
#     base_dataset,
#     roi_width=8,
#     roi_height=8,
#     overlap=0.30,
#     max_rois_per_image=roi_per_image
#     )
#
# cancer_roi_dataset = roi_dataset.get_cancer_samples()
# full_roi_dataset = roi_dataset + cancer_roi_dataset
#
# # {(0, 1): 96.0, (0, 5): 96.0, (0, 6): 135.7645019878171, (1, 6): 96.0, (1, 2): 96.0, (1, 5): 135.7645019878171, (5, 6): 96.0, (6, 7): 96.0, (6, 10): 242.9588442514493, (2, 3): 96.0, (2, 7): 96.0, (3, 8): 96.0, (3, 4): 96.0, (3, 9): 135.7645019878171, (7, 8): 96.0, (7, 10): 205.73040611440985, (8, 4): 135.7645019878171, (8, 9): 96.0, (8, 10): 209.88806540630173, (4, 9): 96.0}
#
#
# patient_labels = []
# for r in full_roi_dataset:
#     patient_labels.append(r["ct_folder"])
# patient_labels = list(set(patient_labels))  # list of patient id's
# new_patient_labels = []
# for patient_label in patient_labels:
#     new_patient_labels.append(patient_label.replace("-", "_"))  # change CT-009 to CT_009
# patientIndex_newPatientIndex_dict = dict(zip(patient_labels, new_patient_labels))  # {CT-009:CT_009,CT-010:CT_010..}
#
# patient_graphs = {}
#
# for patient_label in patient_labels:
#     graph_name = f"spatial_knn_network_{patientIndex_newPatientIndex_dict[patient_label]}"
#     #exec(graph_name + "= nx.Graph()")  # creates n empty graphs, n-no of patients
#     patient_graphs[graph_name] = nx.Graph()
#
#     patient_wise_rois = []
#     for r in full_roi_dataset:  # takes the first non cancer roi
#         if patientIndex_newPatientIndex_dict[r["ct_folder"]] == patientIndex_newPatientIndex_dict[patient_label]:
#             patient_wise_rois.append(r)  # collect the roi's corresponding to 1 patient - list of dict
#
#     edge_list = []
#
#     index_ = -1
#     for roi in patient_wise_rois:  # takes the first non-cancer roi for 1 patient
#         index_ += 1  # add index as the node and rest as node attributes
#         # exec(
#         #     graph_name + ".add_node(index_, r_index=roi['index'], image=roi['image'], coordinates=roi['coordinates'], neighbors=roi['neighbors'], time_point=roi['time_point'], ct_folder=roi['ct_folder'], case_type=roi['case_type'])"
#         #     )
#         patient_graphs[graph_name].add_node(index_, r_index=roi['index'], image=roi['image'], coordinates=roi['coordinates'], neighbors=roi['neighbors'], time_point=roi['time_point'], ct_folder=roi['ct_folder'], case_type=roi['case_type'])
#
#         list_of_edges = []
#         list_of_neighs = []
#         for neighs in roi['neighbors']:  # picks each neighbor(dict) from the list
#             list_of_neighs.append(list(neighs.keys())[0])  # store the index of the neighbor
#             edge_tuple = tuple(sorted([index_, list(neighs.keys())[0]]))  # (0,3)
#             list_of_edges.append(edge_tuple)
#         list_of_edges = list(set(list_of_edges))  # [(0,1),(0,4)]
#
#         dict_of_edge_attributes = {}
#         list_of_edge_attributes = []
#         for edge in list_of_edges:
#             if edge[0] == index_:  # For later use to extract distance
#                 n = edge[1]
#             else:
#                 n = edge[0]
#
#             for neighs2 in roi["neighbors"]:  # picks each neighbor(dict) from the list
#                 if list(neighs2.keys())[0] == n:  # pick up the index of the neighbors
#                     dist = neighs2[n]["distance"]
#             dict_of_edge_attributes[edge] = dist
#             list_of_edge_attributes.append(dist)
#
#             #exec(graph_name + ".add_edge(edge[0],edge[1], spatial_distance=dist)")
#             patient_graphs[graph_name].add_edge(edge[0],edge[1], spatial_distance=dist)
#
#     #draw_network_graph(patient_graphs[graph_name],patient_label)
#
# """
# 10 - 10,10,0.05
# 20 - 10,10,0.25
# 30 - 8,8,0.25
# 40 - 8,8,0.30
# 50 - 8,8,0.40
# 60 - 5,5,0
# """
