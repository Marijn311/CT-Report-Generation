import os
import SimpleITK as sitk
import math
import numpy as np
import pandas as pd
from totalsegmentator.python_api import totalsegmentator # pip3 install git+https://github.com/wasserth/TotalSegmentator.git
BACKGROUND_VALUE = -1000.0 # Float value that is used to fill the background and occluded area of the image. -1000 is the Hounsfield unit for air

"""
This script takes CT images and adds a random combination of artificial abnormalities. 

There are 3 things that require user action:
1. You need to copy a normal CT dataset and rename the copy to something logical like "CT_images_artificial_abnormalities" since this script will overwrite the images.	
2. DATA_PATH: the path to the new copied dataset
3. INPUT_FILE_NAME: the name of the original CT file which will be augmented.
"""

DATASET_PATH = r"Z:\geselecteerd\AA_Marijn_processed\CT_images_3_tasks"
INPUT_FILE_NAME = 'ct_thorax.nrrd'

if __name__ == "__main__": # This is needed to prevent a .freeze() and threading error

    def potentially_mirror_image(image):
        # Chose a random integer, either 0 or 1
        random_int = np.random.randint(2)
        
        # Mirror the array along the sagittal plane if the random integer is 1
        if random_int == 1:
            image_array = sitk.GetArrayFromImage(image)
            mirrored_array = np.flip(image_array, axis=2)
            mirrored_image = sitk.GetImageFromArray(mirrored_array)
            mirrored_image.CopyInformation(image)
            image = mirrored_image
        
        # Determine the corresponding (multihot) label
        label = [random_int]
        return image, label


    def potentially_rotate_image(image):
        # Define the possible rotations in degrees
        rotation_options = [-90, -45, 0, 45, 90]
        
        # Chose a random integer in length of the rotation_options list
        rotation_index = np.random.choice(len(rotation_options))
        
        # Get the rotation angle associated with the random integer
        degree_rotation = rotation_options[rotation_index]
        
        # Convert the angle to radians
        radian_rotation = degree_rotation * (math.pi / 180.0)

        # Determine the center of rotation (in physical coordinates)
        size = image.GetSize()
        rotation_center = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        rotation_center_physical = image.TransformContinuousIndexToPhysicalPoint(rotation_center)
        
        # Define the transformation (rotation around the z-axis)
        transform = sitk.VersorTransform((0,0,1), radian_rotation)
        transform.SetCenter(rotation_center_physical)
        
        # Create a resampler that incorporates the transformation
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(size)
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin([0, 0, 0])
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(BACKGROUND_VALUE) 
        resampler.SetTransform(transform)

        # Perform the resampling operation
        rotated_image = resampler.Execute(image)
        rotated_image.SetOrigin(origin)
        
        # Determine the corresponding multihot label
        label = np.zeros(len(rotation_options))
        label[rotation_index] = 1
 
        return rotated_image, label

    def potentially_occlude_image(image, image_path):
        # Define path where to save the lobe masks
        mask_path = os.path.join(hospital_folder_path, patient_folder, 'lobe_occlusion_mask.nrrd') 

        intermediate_image_nii_path = image_path.replace(".nrrd", ".nii") # Needed intermediate step, since TotalSegmentator does not accept nrrd files
        intermediate_mask_nii_path = mask_path.replace(".nrrd", ".nii") # This has to be a nii file as well. We will rename it to nrrd further down in the code.

        # Save the image as nii file, needed for the segmentation
        sitk.WriteImage(image, intermediate_image_nii_path)
            
        # Randomly select a lobe to occlude                    
        lobe_options = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
        lobe_index = np.random.choice(len(lobe_options))
        
        # Get the chosen lobe
        lobe = lobe_options[lobe_index]
        
        # Segment the chosen lung lobes 
        totalsegmentator(intermediate_image_nii_path, # If you get a NoneType split error: this has to do with numpy version. Read the documentation in the total_segmentor.py file.
                        intermediate_mask_nii_path, 
                        fast = True, 
                        ml = True, 
                        device = "gpu", 
                        preview = False,  
                        roi_subset = [lobe])
        
        
        # Open the image and mask 
        mask = sitk.ReadImage(intermediate_mask_nii_path) 
        mask_array = sitk.GetArrayFromImage(mask)
        image_array = sitk.GetArrayFromImage(image)

        # Apply the mask to the image
        image_array[mask_array>0] = BACKGROUND_VALUE # Somehow the positive values in the mask are 14 and not 1 so we do >0 instead of ==1
        
        # Convert the mask image array back to an image object 
        masked_image = sitk.GetImageFromArray(image_array)
        masked_image.CopyInformation(image)

        # Determine the corresponding multihot label
        label = np.zeros(len(lobe_options))
        label[lobe_index] = 1

        return masked_image, label

    # Loop over the dataset and introduce artificial abnormalities and save the corresponding labels
    for root, hospital_folders, _ in os.walk(DATASET_PATH): 
        for hospital_folder in hospital_folders: 
            hospital_folder_path = os.path.join(root, hospital_folder)
            labels_path = os.path.join(hospital_folder_path, 'labels.xlsx')
            labels = pd.read_excel(labels_path)
            for _, patient_folders, _ in os.walk(hospital_folder_path):
                for patient_folder in patient_folders:         
                    
                    # Define paths
                    image_path = os.path.join(hospital_folder_path, patient_folder, INPUT_FILE_NAME) 
                    assert os.path.exists(image_path), f"image_path does not exist: {image_path}"
                    image = sitk.ReadImage(image_path)
                    image = sitk.DICOMOrient(image, 'LPS')
                    
                    # Potentially occlude the image
                    image, occlusion_label = potentially_occlude_image(image, image_path)
                    
                    # Potentially mirror the image
                    image, mirror_label = potentially_mirror_image(image)
                    
                    # Set the origin: This should remanin below the occlusion but above rotation
                    origin = image.GetOrigin()
                    image.SetOrigin((0,0,0))
                    
                    # Potentially rotate the image
                    image, rotation_label = potentially_rotate_image(image)
                    
                    # Save the resulting image
                    output_image_path = image_path.replace(".nrrd", "_3_abnormalities.nrrd")
                    sitk.WriteImage(image, output_image_path)
                    
                    
                    # Save corresponding label
                    # labels are formatted as [mirrored, all rotation options, all lobe options]
                    new_label = [mirror_label, rotation_label, occlusion_label]
                    new_label = [item for sublist in new_label for item in sublist] 
                    new_label = [int(i) for i in new_label]
                    
                    # Overwrite the existing label for the patient in question
                    labels.loc[labels['PseudoID'] == patient_folder, 'MH_diseases'] = [new_label]
                    print(f"Saved an output file for {patient_folder}")
            # Save the labels excel file
            labels.to_excel(labels_path, index=False)
        break             