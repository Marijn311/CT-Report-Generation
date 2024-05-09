import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import random_split
import numpy as np
import pandas as pd
import os
from config import *
import SimpleITK as sitk
from dataset_preprocessing.ct_vol_preprocessing import preprocess_ct_volume
import einops
from utils import show_getitem_samples


# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

class dataset(torch.utils.data.Dataset):
    """This class represents a dataset that contains pairs of CT images and their corresponding report.

    Attributes:
        hospital_dataset_lengths (list): A list containing the lengths of all hospital datasets.
        global_len (int): The total number of image/report pairs in the entire dataset.
        
    Methods:
        __len__(): Return the length of the full dataset.
        get_position(global_index, dataset_sizes): Convert a global index to a specific hospital dataset and an index in that dataset.
        __getitem__(index): Returns one image and its label.
        
    """
    def __init__(self):
        super().__init__()
   
        # Initialize the attributes
        self.hospital_dataset_lengths = []
        self.global_len = 0
        
        # Used as intermediate storage
        report_pseudo_ids = []
        image_pseudo_ids = []
        
        # Load the reports
        for dataset_name in HOSPITAL_NAMES:
            labels_path=os.path.join(IMAGE_DIR, dataset_name, 'labels.xlsx')
            labels = pd.read_excel(labels_path)
            
            # Count the number of reports by extracting all the pseudo IDs
            nr_labels = len(labels.index) 
            self.hospital_dataset_lengths.append((dataset_name, nr_labels))
            self.global_len += nr_labels 
            report_ids = labels['PseudoID'].tolist() 
            report_pseudo_ids.extend(report_ids)
            
            # Count the number of images by extracting all the pseudo IDs
            for _, folders, _ in os.walk(os.path.join(IMAGE_DIR, dataset_name)):
                for folder in folders:
                    image_pseudo_ids.append(folder)
        
        # Make sure that all reports have an image and all images have a report
        for report_pseudo_id in report_pseudo_ids:
            if report_pseudo_id not in image_pseudo_ids:
                print(f"Report with PseudoID {report_pseudo_id} does not have an image.")
        for image_pseudo_id in image_pseudo_ids:
            if image_pseudo_id not in report_pseudo_ids:
                print(f"Image with PseudoID {image_pseudo_id} does not have a report.")
        assert len(report_pseudo_ids) == len(image_pseudo_ids), f"There are {len(report_pseudo_ids)} reports and {len(image_pseudo_ids)} images. This should be equal."
        print(f"{self.global_len} image/labels pairs where loaded.")
        
        
    def __len__(self):
        "Return the length of the full dataset."
        return self.global_len
    
    
    def get_position(self, global_index, dataset_sizes):
        "Convert a global index to a specific hospital dataset and an index in that dataset"
        current_index = global_index
        for count, dataset_size in enumerate(dataset_sizes):
            if current_index < dataset_size:
                return count, current_index
            current_index -= dataset_size
    
    
    def __getitem__(self, index):
        """Returns one image and its label."""
        
        # Convert the global index to a specific hospital dataset and an index in that dataset
        dataset_sizes = [dataset_length[1] for dataset_length in self.hospital_dataset_lengths]
        dataset_index, position_in_dataset = self.get_position(index, dataset_sizes)
        dataset_name = self.hospital_dataset_lengths[dataset_index][0]
        dataset_folder_path = os.path.join(IMAGE_DIR, dataset_name)
        subfolders = [f.path for f in os.scandir(dataset_folder_path) if f.is_dir()] 
        subfolder = subfolders[position_in_dataset]
        subfolder_name = os.path.basename(subfolder)
        input_path=os.path.join(IMAGE_DIR, dataset_name, subfolder_name, IMAGE_FILE_NAME)
        
        
        # Load and preprocess the image
        if FILE_TYPE == 'image':
            # Load the image and put it in the default orientation
            image = sitk.ReadImage(input_path)
            image = sitk.DICOMOrient(image, 'LPS') 
           
            # Make the voxel spacing isotropic by calculating the new size of the image and resampling the image to this new size
            spacing = image.GetSpacing()
            new_size = [int(image.GetSize()[i] * spacing[i]) for i in range(3)]
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing((1.0, 1.0, 1.0))
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetDefaultPixelValue(BACKGROUND_VALUE) 
            resampled_image = resampler.Execute(image)
            resampled_image_array = sitk.GetArrayFromImage(resampled_image)

            # Preprocess the image
            image_processed = preprocess_ct_volume(ctvol=resampled_image_array, pixel_bounds=[BACKGROUND_VALUE, 1300], data_augment=DATA_AUGMENT, num_channels=NUM_CHANNELS, max_slices=IMAGE_SIZE[0], max_side_length=IMAGE_SIZE[1], center_on_imagenet_mean=CENTER_ON_IMAGENET_MEAN, padding_value=BACKGROUND_VALUE)
        
        # Load the feature tensor
        if FILE_TYPE == 'feature_tensor':
            image_processed = torch.load(input_path)
            
        # Load the label corresponding to the image / feature tensor
        label_path=os.path.join(IMAGE_DIR, dataset_name, 'labels.xlsx')
        reports_df = pd.read_excel(label_path)
        label_string = reports_df.loc[reports_df['PseudoID'] == subfolder_name, 'MH_diseases'].iloc[0]
        mh_label = np.fromstring(label_string[1:-1], dtype=np.float32, sep=',')
        mh_label = torch.from_numpy(mh_label)
        assert mh_label.shape[0] == NUM_CLASSES, f"Expected {NUM_CLASSES} labels per sample in the label file because that is the number of classes as defined by NUM_CLASSES. Instead there are {mh_label.shape[0]} labels per sample."
        
        # Show some examples of the data
        if SHOW_DATA_EXAMPLES == True:
            show_getitem_samples(image_processed, subfolder_name, mh_label)
                    
        return image_processed, mh_label


class data_module(pl.LightningDataModule):
    """LightningDataModule subclass for handling the data module of the model.

    This class provides the necessary methods to set up the data module, including splitting the dataset into train, validation, and test sets,
    and creating dataloaders for each set.

    Attributes:
        entire_ds (Dataset): The entire dataset.
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        test_ds (Dataset): The test dataset.

    Methods:
        setup(stage): Method to set up the data module.
        train_dataloader(): Method to create a dataloader for the training dataset.
        val_dataloader(): Method to create a dataloader for the validation dataset.
        test_dataloader(): Method to create a dataloader for the test dataset.
    """
    def __init__(self):
        super().__init__()

    def setup(self, stage):
        self.entire_ds = dataset() 
        generator1 = torch.Generator().manual_seed(42) # To ensure reproducibility in the train/val/test split
        self.train_ds, self.val_ds, self.test_ds = random_split(self.entire_ds, DATA_SPLIT, generator=generator1)
       
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=True)
    