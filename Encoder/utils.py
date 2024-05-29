import torch
from config import *
import os
import einops
from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Set seeds
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

def save_feature_maps(cloned_features):
    """"This function is used to save the feature tensors (the state of the images after the last convolutional layer in the CNN) to a .pt file. 
    
    Required actions:
    The function requires some specific settings to function propperly.
    Follow the instructions precisely, else you might overwrite some existing files with corrupted files.
    
    1. Set stage to validate in config.py
    2. Set the nr of epochs to 1 in config.py 
    3. Set batchsize to 1 in config.py
    4. Change self.val_ds to self.entire_ds in the validation dataloader in dataset.py, make sure Shuffle is set to False
    5. Set DATA_AUGMENT to False in config.py
    6. Set the IMAGE_FILE_NAME in config.py to be the .nrrd file that you want to save the features of. This should NOT be .pt file.
    7. Set the FILE_NAME in the code below, the file_name should be the name of the .pt file under which the features will be saved.
    8. Check if the .pt file you want to create already exists in any folder. If that is the case we don't want to run this script because it will save the first feature tensor in the first folder WHERE THIS FILE DOES NOT EXIST. This is not necessarily the first folder in the dataset. So if any of the files already exists, we have to delete them first.             
    9. Set SHOW_DATA_EXAMPLES to True in config.py, and verify that the first image that is loaded is the first image of the first hospital etc (so verify that the images are not shuffled)
    10. If everything is set up correctly, turn off the SHOW_DATA_EXAMPLES again and run the code. It will save the features to a .pt file in the correct folder.
    11. Dont forget to undo these changes when you are done with making .pt files 
    """ 

    FILENAME = 'med_net_features_ct_thorax_lobe_occlusion_254x260x260.pt' 

    # Check the assumptions where possible
    assert BATCH_SIZE == 1, "BATCH_SIZE should be 1 to save the features to a .pt file" 
    assert NUM_EPOCHS == 1, "NUM_EPOCHS should be 1 to save the features to a .pt file"
    assert DATA_AUGMENT == False, "DATA_AUGMENT should be False to save the features to a .pt file"
    assert STAGE == "validate", "STAGE should be validate, because this will load the weights and process all images"

    # Create a pop message informing the user that it is important to follow the instructions
    root = tk.Tk()
    root.withdraw() 
    messagebox.showinfo("ATTENTION", "Make sure that you have followed the instructions in the save_feature_maps function in utils.py.")
    root.mainloop()

    # Loop over all patients and save the features to the first empty folder
    for root, hospital_folders, _ in os.walk(IMAGE_DIR): 
        for hospital_folder in hospital_folders: 
            hospital_folder_path = os.path.join(root, hospital_folder)
            for _, patient_folders, _ in os.walk(hospital_folder_path):
                for patient_folder in patient_folders: 
                    
                    # Define path
                    pt_path = os.path.join(hospital_folder_path, patient_folder, FILENAME)

                    # Find the first empty folder and save the features there
                    if not os.path.exists(pt_path):
                        torch.save(cloned_features, pt_path)
                        print(f"saved features to {pt_path}")
                        return None                             # This return statement makes sure that the function stops after the first file is saved. We only save one file per call of the function
        breakpoint()                                            # This breakpoint makes sure that the function stops after all the hospitals are processed and that we do not loop over other folders that are in the main folder       





 
def save_encoded_images(x):
    """"This function is used to save the encoded images (the state of the images before the final linear layer in the classifier) to a .pt file. 
    
    Required actions:
    The function requires some specific settings to function propperly.
    Follow the instructions precisely, else you might overwrite some existing files with corrupted files.
    
    1. Set stage to validate in config.py
    2. Set the nr of epochs to 1 in config.py 
    3. Set batchsize to 1 in config.py
    4. Change self.val_ds to self.entire_ds in the validation dataloader in dataset.py. Make sure Shuffle is set to False
    5. Set DATA_AUGMENT to False in config.py
    6. Set the IMAGE_FILE_NAME in config.py to be the feature map file that you want to save the features of. This should be .pt file and it will also create a .pt file.
    7. Set the FILE_NAME in the code below, the file_name should be the name of the .pt file under which the encoded image will be saved.
    8. Check if the .pt file you want to create already exists in any folder. If that is the case we don't want to run this script because it will save the first feature tensor in the first folder WHERE THIS FILE DOES NOT EXIST. This is not necessarily the first folder in the dataset. So if any of the files already exists, we have to delete them first.             
    9. Set SHOW_DATA_EXAMPLES to True in config.py, and verify that the first image that is loaded is the first image of the first hospital etc (so verify that the images are not shuffled)
    10. If everything is set up correctly, turn off the SHOW_DATA_EXAMPLES again and run the code. It will save the features to a .pt file in the correct folder.
    11. Dont forget to undo these changes when you are done with making .pt files 
    """ 

    FILENAME = 'ct_net_3d_convs_mirroredness_1_a.pt' 
   
    # Check the settings where possible
    assert BATCH_SIZE == 1, "BATCH_SIZE should be 1 to save the features to a .pt file" 
    assert NUM_EPOCHS == 1, "NUM_EPOCHS should be 1 to save the features to a .pt file"
    assert DATA_AUGMENT == False, "DATA_AUGMENT should be False to save the features to a .pt file"
    assert STAGE == "validate", "STAGE should be validate, because this will load the weights and process all images"

    # Loop over all patients and save the features to the first empty folder
    for root, hospital_folders, _ in os.walk(IMAGE_DIR): 
        for hospital_folder in hospital_folders: 
            hospital_folder_path = os.path.join(root, hospital_folder)
            for _, patient_folders, _ in os.walk(hospital_folder_path):
                for patient_folder in patient_folders:
                    
                    # Define path
                    pt_path = os.path.join(hospital_folder_path, patient_folder, FILENAME)

                    # Find the first empty folder and save the features there
                    if not os.path.exists(pt_path):
                        x = einops.rearrange(x, '1 x -> x') # Remove the singleton dimension for the batch size
                        torch.save(x, pt_path)
                        print(f"saved features to {pt_path}")
                        return None                             # This return statement makes sure that the function stops after the first file is saved. We only save one file per call of the function
        breakpoint()                                            # This breakpoint makes sure that the function stops after all the hospitals are processed and that we do not loop over other folders that are in the main folder





def show_getitem_samples(image_processed, subfolder_name, mh_label):
    """
    Display samples of image slices and a 3D volume rendering of the image.

    Parameters:
    image_processed (torch.Tensor): The processed image tensor.
    subfolder_name (str): The name of the subfolder.
    mh_label (str): The label for the image.

    Returns:
    None
    """

    # Generate 16 evenly spaced slice indexes from the image
    # The shapes/dimensions can differ between the architectures and file types
    # The slice dimension is the one that should be extracted
    if ARCHITECTURE == 'medical_net' and FILE_TYPE == 'image':                          
        slice_idxs = np.linspace(0, image_processed.shape[1]-1, num=16, dtype=int)
    if ARCHITECTURE == 'ct_net':
        slice_idxs = np.linspace(0, image_processed.shape[0]-1, num=16, dtype=int)
        
    # Plot the slices
    fig, axs = plt.subplots(4, 4, figsize=(15, 5))
    for i, idx in enumerate(slice_idxs):
        
        # Again the shapes/dimensions can differ between the architectures and file types
        if ARCHITECTURE == 'medical_net' and FILE_TYPE == 'image':
            axs[i//4, i%4].imshow(image_processed[0,idx,:,:].cpu().detach().numpy(), cmap='gray')
        elif ARCHITECTURE == 'ct_net' and FILE_TYPE == 'image': 
            axs[i//4, i%4].imshow(image_processed[idx,0,:,:].cpu().detach().numpy(), cmap='gray')
        elif ARCHITECTURE == 'medical_net' and FILE_TYPE == 'feature_tensor':
            axs[i//4, i%4].imshow(image_processed[idx,1,1,:,:].cpu().detach().numpy(), cmap='gray')
        elif ARCHITECTURE == 'ct_net' and FILE_TYPE == 'feature_tensor':
            axs[i//4, i%4].imshow(image_processed[idx,0,:,:].cpu().detach().numpy(), cmap='gray')
    
    # Add a PseudoID and label to the plot
    fig.suptitle(f'Patient {subfolder_name} \n, label: {mh_label}')
    plt.show()


    # Show a 3D volume rendering of the image
    if FILE_TYPE == 'image':
        # Again the shapes/dimensions can differ between the architectures
        if ARCHITECTURE == 'ct_net':
            array_3d_to_render = einops.rearrange(image_processed, 'chunks c x y -> (chunks c) x y')
            array_3d_to_render = array_3d_to_render.cpu().detach().numpy()
        if ARCHITECTURE == 'medical_net':
            array_3d_to_render = image_processed[0,:,:,:].cpu().detach().numpy()
       
        
        fig = mlab.figure()       
        scalar_field = mlab.pipeline.scalar_field(array_3d_to_render) 
        volume = mlab.pipeline.volume(scalar_field) # This volume seems to be unused, but it is necessary for the volume rendering to work 
        mlab.axes()
        mlab.orientation_axes()
        mlab.show()