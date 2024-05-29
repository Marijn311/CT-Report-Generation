import os
import SimpleITK as sitk
import numpy as np
from config import BACKGROUND_VALUE

"""
This script applies a mask to the CT images and saves the masked image for each patient in the dataset.
The mask is applied by setting the pixel value of the image to -1000 (air in HoundsField units) where the mask is 0.

There are 4 things that require user input:
1. dataset_path: the path to the dataset
2. ct_filename: the name of the CT file that will be loaded
3. mask_filename: the name of the mask file that will be loaded
4. masked_image_filename: the name of the masked image file taht will be saved
    
"""

DATASET_PATH = r"Z:\geselecteerd\AA_Marijn_processed/CT_images"
CT_FILENAME = 'ct_thorax.nrrd'
MASK_FILENAME = 'lung_mask.nrrd'
MASKED_IMAGE_FILENAME = 'ct_lungs.nrrd'

# Loop over all the images in the dataset
for root, hospital_folders, _ in os.walk(DATASET_PATH):
    for hospital_folder in hospital_folders: 
        hospital_folder_path = os.path.join(root, hospital_folder)
        for _, patient_folders, _ in os.walk(hospital_folder_path):
            for patient_folder in patient_folders: 
                
                # Define paths
                ct_path = os.path.join(hospital_folder_path, patient_folder, CT_FILENAME) 
                mask_path = os.path.join(hospital_folder_path, patient_folder, MASK_FILENAME)
                masked_image_path = os.path.join(hospital_folder_path, patient_folder, MASKED_IMAGE_FILENAME)
                 
                # Check if the masked image already exists
                if not os.path.exists(masked_image_path): 
                    
                    # Open the image and the mask
                    img = sitk.ReadImage(ct_path)
                    img_data = sitk.GetArrayFromImage(img)
                    mask = sitk.ReadImage(mask_path)
                    mask_data = sitk.GetArrayFromImage(mask)
                    
                    # Apply the mask
                    masked_img_data = np.where(mask_data == 0, BACKGROUND_VALUE, img_data)
                    
                    # Save the masked image
                    masked_img = sitk.GetImageFromArray(masked_img_data)
                    masked_img.SetSpacing(img.GetSpacing())
                    masked_img.SetOrigin(img.GetOrigin())
                    masked_img.SetDirection(img.GetDirection())
                    sitk.WriteImage(masked_img, masked_image_path)
                print(f"Saved a masked image for patient: {patient_folder}")
    break 