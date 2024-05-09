import os

"""
This script allows you to rename or delete files for each patient in the dataset.
   
There are 4 things that require user input:
1. dataset_path: the path to the dataset
2. existing_file_name: the name of the file that will be renamed or deleted
3. delete_or_rename: choose 'delete' or 'rename'
4. new_file_name: the new name of the file, this is only used when delete_or_rename is set to 'rename'
     
"""

DATASET_PATH = r"Z:\geselecteerd\AA_Marijn_processed\CT_images_3_tasks"
EXISTING_FILE_NAME = 'med_net_features_ct_thorax_254x260x260.pt'
DELETE_OR_RENAME = 'delete' # Either 'delete' or 'rename'
NEW_FILE_NAME = 'ct_thorax_lobe_deletion.nrrd' # This is not used when "delete" is chosen

# Loop over the dataset and rename or delete the files
for root, hospital_folders, _ in os.walk(DATASET_PATH):
    for hospital_folder in hospital_folders: 
        hospital_folder_path = os.path.join(root, hospital_folder)
        for _, patient_folders, _ in os.walk(hospital_folder_path):
            for patient_folder in patient_folders:
                
                # Define the paths
                existing_path = os.path.join(hospital_folder_path, patient_folder, EXISTING_FILE_NAME)
                new_path = os.path.join(hospital_folder_path, patient_folder, NEW_FILE_NAME)    
                
                assert DELETE_OR_RENAME in ['delete', 'rename'], 'DELETE_OR_RENAME should be set to "delete" or "rename"'
                if DELETE_OR_RENAME == 'rename':
                    if os.path.exists(existing_path): 
                        os.rename(existing_path, new_path)
                    
                if DELETE_OR_RENAME == 'delete':
                    if os.path.exists(existing_path):
                        os.remove(existing_path)
    break




