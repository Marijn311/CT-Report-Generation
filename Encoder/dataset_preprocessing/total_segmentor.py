from totalsegmentator.python_api import totalsegmentator # pip3 install git+https://github.com/wasserth/TotalSegmentator.git
import os
import SimpleITK as sitk
import torch

"""This script allows you to segment anatomical structures in CT images using the TotalSegmentator models.
The TotalSegmentator model requires a different version of numpy than all the other scripts/models.
Therefore, you need to make a different venv for this script.
You can define which structures you want to segment by changing the roi_subset parameter.
Once you have a segmentation mask of an anatomic structure for every image, you can apply the masks to the correspindingimages using the apply_mask.py script.

There are 4 things that require user input:
1. DATA_DIR: the path to the dataset
2. ORIGINAL_FILENAME: the name of the original CT file from which the anatomical structure will be segmented
3. MASK_FILENAME: the name of the file that will contain the segmentation mask
4. ROI_SUBSET: the anatomical structures that we want to segment. This is a list of strings. You can find the possible values at the bottom of the following link: https://github.com/wasserth/TotalSegmentator         
    
"""
    
DATASET_PATH = r"Z:\geselecteerd\AA_Marijn_processed\CT_images"
ORIGINAL_FILENAME = 'CT.nrrd'
MASK_FILENAME = 'lung_mask.nii' # This needs to be a nii file, since the TotalSegmentator does not accept nrrd files
ROI_SUBSET = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "brachiocephalic_vein_left", "brachiocephalic_vein_right"]
          
                        
if __name__ == "__main__": # This is needed to prevent a .freeze() and threading error

    # Check if we have a gpu available
    assert torch.cuda.is_available(), "I strongly recommend using a GPU for this script, else it will take very long."
    print(f"Availablke GPU: {torch.cuda.get_device_name(0)}")
        
    # Loop over all the images in the dataset
    for root, hospital_folders, _ in os.walk(DATASET_PATH): 
        for hospital_folder in hospital_folders: 
            hospital_folder_path = os.path.join(root, hospital_folder)
            for _, patient_folders, _ in os.walk(hospital_folder_path):
                for patient_folder in patient_folders: 
                    
                    # Define paths
                    original_image_path = os.path.join(hospital_folder_path, patient_folder, ORIGINAL_FILENAME)
                    intermediate_image_nii_path = original_image_path.replace(".nrrd", ".nii") # Needed intermediate step, since TotalSegmentator does not accept nrrd files
                    intermediate_mask_nii_path = os.path.join(hospital_folder_path, patient_folder, MASK_FILENAME)

                    # Check if the segmentation already exists
                    if not os.path.exists(intermediate_mask_nii_path):
                        print(f"Started working on {patient_folder}")
                
                        # Convert the original image from .nrrd to .nii because the totalsegmentator does not accept nrrd
                        nrrd_img = sitk.ReadImage(original_image_path)
                        sitk.WriteImage(nrrd_img, intermediate_image_nii_path)
                        
                        # Segment an anatomical structure(s)
                        totalsegmentator(intermediate_image_nii_path, # If you get a NoneType split error: this has to do with numpy version. Read the documentation at the top of this script!
                                        intermediate_mask_nii_path, 
                                        fast=True, 
                                        ml=True,
                                        device="gpu", 
                                        preview=False,  # Using preview=True gives an error for me
                                        roi_subset=ROI_SUBSET)
                        
                        # Convert the mask back to nrrd
                        mask_img = sitk.ReadImage(intermediate_mask_nii_path)
                        sitk.WriteImage(mask_img, intermediate_mask_nii_path.replace(".nii", ".nrrd"))
                        
                        # Remove the intermediate nii files
                        os.remove(intermediate_image_nii_path) 
                        os.remove(intermediate_mask_nii_path) 
        break                






