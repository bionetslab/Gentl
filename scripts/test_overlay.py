import os

import matplotlib.pyplot as plt


def process_folder(folder_path, time_point):
    for ct_folder in os.listdir(folder_path):
        ct_path = os.path.join(folder_path, ct_folder)
        if not os.path.isdir(ct_path):
            continue

        for case_type in ['Lesion', 'Control']:
            case_path = os.path.join(ct_path, case_type)
            if not os.path.isdir(case_path):
                continue

            for file in os.listdir(case_path):
                if file.endswith('.png') and case_path == "Lesion":
                    mask_cancer_ = os.path.join(case_path, file)
                elif file.endswith('.jpg') and case_path == "Lesion":  # image with background pixel as 0
                    background_image_ = os.path.join(case_path, file)
                elif file.endswith('.png') and case_path == "Control":
                    mask_control_ = os.path.join(case_path, file)
            mask_cancer = mask_cancer_
            background_image = background_image_
            mask_control = mask_control_


root_dir = "../data/original/Al-Bladder Cancer"

for main_folder in os.listdir(root_dir):
    main_path = os.path.join(root_dir, main_folder)
    # print("Main Path: ", main_path)
    if not os.path.isdir(main_path) or main_folder == 'Redo':
        continue  # continue if redo folder or if its not a directory
    else:
        process_folder(main_path, main_folder)
