import os
import matplotlib.pyplot as plt
try:
    from PIL import Image
except ImportError:
    import Image


root_dir = "../data/original/Al-Bladder Cancer"
output_dir = "../data/overlay_images/"

for main_folder in os.listdir(root_dir):
    main_path = os.path.join(root_dir, main_folder)
    if not os.path.isdir(main_path) or main_folder == 'Redo':
        continue  # continue if redo folder or if its not a directory
    else:
        # process_folder(main_path, main_folder)
        for ct_folder in os.listdir(main_path):
            ct_path = os.path.join(main_path, ct_folder)
            if not os.path.isdir(ct_path):
                continue
            paths = []
            for case_type in ['Lesion', 'Control']:
                case_path = os.path.join(ct_path, case_type)
                if not os.path.isdir(case_path):
                    continue
                for file in os.listdir(case_path):
                    if file.endswith('.png') and case_type == "Lesion":
                        mask_cancer_ = os.path.join(case_path, file)
                        paths.append(mask_cancer_)
                    elif file.endswith('.jpg') and case_type == "Lesion":  # image with background pixel as 0
                        background_image_ = os.path.join(case_path, file)
                        paths.append(background_image_)
                    elif file.endswith('.png') and case_type == "Control":
                        mask_control_ = os.path.join(case_path, file)
                        paths.append(mask_control_)

            # Open the images
            cancer_mask = Image.open(paths[1]).convert("RGBA")  # cancer mask
            background_image = Image.open(paths[0]).convert("RGBA")  # background image
            normal_mask = Image.open(paths[2]).convert("RGBA")  # normal mask

            # Blend the cancer mask with an empty transparent image to create a semi-transparent overlay
            cancer_overlay = Image.blend(Image.new("RGBA", cancer_mask.size, (0, 0, 0, 0)), cancer_mask, 0.5)

            # Blend the normal mask with an empty transparent image
            normal_overlay = Image.blend(Image.new("RGBA", normal_mask.size, (0, 0, 0, 0)), normal_mask, 0.5)

            # Overlay the blended masks on the original background image
            combined_overlay = Image.alpha_composite(background_image, cancer_overlay)
            final_image = Image.alpha_composite(combined_overlay, normal_overlay)

            # Save the final image
            final_image.save(f"{output_dir}{ct_folder}.png", "PNG")
