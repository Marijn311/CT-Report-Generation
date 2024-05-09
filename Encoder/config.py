import torch

#Set seeds
RANDOM_SEED = 21
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

"""This is the config file. This file contains nearly all hyperparameters and settings that you might want to change.
Hyperparameters/settings are recognisable by the capital letters.
These Hyperparameters/settings are imported into nearly all other scripts in this project.
These other scripts should never alter the values of the hyperparameters/settings.
By importing the hyperparameters/settings from this file directly, 
there is no need to constantly pass them as arguments to every function or class that needs them."""


# Model/Trainer settings
STAGE = "fit"                   # Either "fit", "validate", or "test"
ARCHITECTURE = "ct_net"             # Either "ct_net" or "medical_net"
CLASSIFIER = "convs_3d"             # Either "convs_3d", "attention_pooling", or "transformer_attention"
BATCH_SIZE = 1 
GRAD_ACCUM = 1 
NUM_EPOCHS = 1 
LEARNING_RATE = 1e-4 
ACCELERATOR = "gpu"
DEVICES = 1 
PRECISION = 32 
NUM_WORKERS = 0 


# Plotting and printing settings
SHOW_PREDICTIONS = False            # Show the predicted labels and the ground truth. Print metrics
SHOW_DATA_EXAMPLES = False          # Show the preprocessed images in 2D and 3D, and the labels before they are fed into the model


# Checkpoint paths
CKPT_PATH_TO_LOAD = r"Z:\geselecteerd\ZZ_model_checkpoints\CT_net\convs_3d\mirroredness 1 a\model-step=579-val_loss=0.11.ckpt"      # Checkpoint to load for validation or testing
CKPT_PATH_TO_SAVE = rf"Z:\geselecteerd\ZZ_model_checkpoints\{ARCHITECTURE}\{CLASSIFIER}"                                            # Path where checkpoints are saved during trainining


# Dataset settings
"""Required dataset structure:

[IMAGE_DIR]
│   [hospital_name_1]
│   │   [hospital_name_1]_[pseudo_id_1]
│   │   │   [IMAGE_FILE_NAME]
│   │   [hospital_name_1]_[pseudo_id_2]
|   |   │   [IMAGE_FILE_NAME]
|   |   ...
|   |   labels.xlsx
|   |   verslagen.docs
│   [hospital_name_2]
|   |   [hospital_name_2]_[pseudo_id_1]
|   |   │   [IMAGE_FILE_NAME]
|   |   [hospital_name_2]_[pseudo_id_2]
|   |   │   [IMAGE_FILE_NAME]
|   |   ...
|   |   labels.xlsx
|   |   verslagen.docs
|   ...
|   all_labels.xlsx
|   all_reports.docs
|   SARLE_dataset.xlsx

"""

# HOSPITAL_NAMES = ['AMPH', 'ISAL', 'LUMC', 'MAXI', 'RADB', 'UMG1', 'UMG2', 'VUMC', 'ZUYD']
# CLASS_NAMES = ['pulmonary_nodule']
# NUM_CLASSES = len(CLASS_NAMES) 
# ARTIFICIAL_ABNORMALITY = 'none'
# POSITIVE_WEIGHTS = torch.tensor([0.625]) 
# DATA_SPLIT = [364,50,50] # Train/val/test 
# IMAGE_FILE_NAME = 'ct_lungs.nrrd'
# SAVE_FEATURE_MAPS = True # Whether to save the feature maps that are produced by the CNN to a .pt file
# SAVE_ENCODED_IMAGES = True # Whether to save the states of the last linear layer in the classifier to a .pt file
# BACKGROUND_VALUE = -1000 # -1000 is air in Hounsfield units
# DATA_AUGMENT = False 

# HOSPITAL_NAMES = ['AMPH', 'ISAL', 'LUMC', 'MAXI', 'RADB', 'UMG1', 'UMG2', 'VUMC', 'ZUYD']
# CLASS_NAMES = ['pleural_effusion'] 
# NUM_CLASSES = len(CLASS_NAMES) 
# ARTIFICIAL_ABNORMALITY = 'none'
# POSITIVE_WEIGHTS = torch.tensor([11.5])
# DATA_SPLIT = [364,50,50] # Train/val/test 
# IMAGE_FILE_NAME = 'ct_thorax.nrrd'
# SAVE_FEATURE_MAPS = True # Whether to save the feature maps that are produced by the CNN to a .pt file
# SAVE_ENCODED_IMAGES = True # Whether to save the states of the last linear layer in the classifier to a .pt file
# BACKGROUND_VALUE = -1000 # -1000 is air in Hounsfield units
# DATA_AUGMENT = False 

# HOSPITAL_NAMES = ['AMPH_0_degree', 'AMPH_45_degree', 'AMPH_90_degree', 'AMPH_minus_45_degree', 'AMPH_minus_90_degree', 'ISAL_0_degree', 'ISAL_45_degree', 'ISAL_90_degree', 'ISAL_minus_45_degree', 'ISAL_minus_90_degree', 'LUMC_0_degree', 'LUMC_45_degree', 'LUMC_90_degree', 'LUMC_minus_45_degree', 'LUMC_minus_90_degree', 'MAXI_0_degree', 'MAXI_45_degree', 'MAXI_90_degree', 'MAXI_minus_45_degree', 'MAXI_minus_90_degree', 'RADB_0_degree', 'RADB_45_degree', 'RADB_90_degree', 'RADB_minus_45_degree', 'RADB_minus_90_degree', 'UMG1_0_degree', 'UMG1_45_degree', 'UMG1_90_degree', 'UMG1_minus_45_degree', 'UMG1_minus_90_degree', 'UMG2_0_degree', 'UMG2_45_degree', 'UMG2_90_degree', 'UMG2_minus_45_degree', 'UMG2_minus_90_degree', 'VUMC_0_degree', 'VUMC_45_degree', 'VUMC_90_degree', 'VUMC_minus_45_degree', 'VUMC_minus_90_degree', 'ZUYD_0_degree', 'ZUYD_45_degree', 'ZUYD_90_degree', 'ZUYD_minus_45_degree', 'ZUYD_minus_90_degree']
# IMAGE_DIR = r"Z:\geselecteerd\AA_Marijn_processed\CT_images_rotation"
# CLASS_NAMES = ['-90', '-45', '0', '45', '90']
# NUM_CLASSES = len(CLASS_NAMES) 
# POSITIVE_WEIGHTS = torch.tensor([4, 4, 4, 4, 4])
# DATA_SPLIT = [1820,250,250] # Train/val/test 
# IMAGE_FILE_NAME = 'ct_thorax_rotated.nrrd'
# SAVE_FEATURE_MAPS = True # Whether to save the feature maps that are produced by the CNN to a .pt file
# SAVE_ENCODED_IMAGES = True # Whether to save the states of the last linear layer in the classifier to a .pt file
# BACKGROUND_VALUE = -1000 # -1000 is air in Hounsfield units
# DATA_AUGMENT = False 

# HOSPITAL_NAMES = ['AMPH_left_lower', 'AMPH_left_upper', 'AMPH_right_lower', 'AMPH_right_middle', 'AMPH_right_upper', 'ISAL_left_lower', 'ISAL_left_upper', 'ISAL_right_lower', 'ISAL_right_middle', 'ISAL_right_upper', 'LUMC_left_lower', 'LUMC_left_upper', 'LUMC_right_lower', 'LUMC_right_middle', 'LUMC_right_upper', 'MAXI_left_lower', 'MAXI_left_upper', 'MAXI_right_lower', 'MAXI_right_middle', 'MAXI_right_upper', 'RADB_left_lower', 'RADB_left_upper', 'RADB_right_lower', 'RADB_right_middle', 'RADB_right_upper', 'UMG1_left_lower', 'UMG1_left_upper', 'UMG1_right_lower', 'UMG1_right_middle', 'UMG1_right_upper', 'UMG2_left_lower', 'UMG2_left_upper', 'UMG2_right_lower', 'UMG2_right_middle', 'UMG2_right_upper', 'VUMC_left_lower', 'VUMC_left_upper', 'VUMC_right_lower', 'VUMC_right_middle', 'VUMC_right_upper', 'ZUYD_left_lower', 'ZUYD_left_upper', 'ZUYD_right_lower', 'ZUYD_right_middle', 'ZUYD_right_upper']
# IMAGE_DIR = r"Z:\geselecteerd\AA_Marijn_processed\CT_images_lobe_deletion"
# CLASS_NAMES = ["left_lower", "left_upper", "right_lower", "right_middle", "right_upper"]
# NUM_CLASSES = len(CLASS_NAMES) 
# POSITIVE_WEIGHTS = torch.tensor([4, 4, 4, 4, 4])
# DATA_SPLIT = [1820,250,250] # Train/val/test 
# IMAGE_FILE_NAME = 'ct_thorax_lobe_deletion.nrrd'
# SAVE_FEATURE_MAPS = True # Whether to save the feature maps that are produced by the CNN to a .pt file
# SAVE_ENCODED_IMAGES = True # Whether to save the states of the last linear layer in the classifier to a .pt file
# BACKGROUND_VALUE = -1000 # -1000 is air in Hounsfield units
# DATA_AUGMENT = False 

HOSPITAL_NAMES = ['AMPH', 'ISAL', 'LUMC', 'MAXI', 'RADB', 'UMG1', 'UMG2', 'VUMC', 'ZUYD']
IMAGE_DIR = r"Z:\geselecteerd\AA_Marijn_processed\CT_images_black_and_mirrored"
CLASS_NAMES = ["mirroredness"]
NUM_CLASSES = len(CLASS_NAMES) 
POSITIVE_WEIGHTS = torch.tensor([1])
DATA_SPLIT = [364,50,50] # Train/val/test 
IMAGE_FILE_NAME = 'ct_half_mirrored.nrrd'
SAVE_FEATURE_MAPS = True # Whether to save the feature maps that are produced by the CNN to a .pt file
SAVE_ENCODED_IMAGES = True # Whether to save the states of the last linear layer in the classifier to a .pt file
BACKGROUND_VALUE = -1000 # -1000 is air in Hounsfield units
DATA_AUGMENT = False 


# Define whether the input is an image  or a feature tensor (which should be passed to the transformer
if IMAGE_FILE_NAME.split('.')[-1] == 'nrrd':
    FILE_TYPE = 'image'                         # If the input is an image it will be passed to a CNN+classifier combo or it will be passed to a CNN and the resulting feature tensor will be saved to a .pt file
elif IMAGE_FILE_NAME.split('.')[-1] == 'pt':
    FILE_TYPE = 'feature_tensor'                # If the input is a feature tensor it will be passed to a classifier
else:
    assert False, "The file type of the given input is not supported. Only .nrrd files are support for images and only .pt files are supported for feature tensors."


# Set image size for the model architectures
if ARCHITECTURE == "ct_net" and FILE_TYPE == "image":
    IMAGE_SIZE = [390, 400, 400] # This is the shape the model desires. Format: [slices, side, side]. The inputted image will be rescaled/cropped/padded during preprocessing to match this size.
if ARCHITECTURE == "medical_net" and FILE_TYPE == "image":
    IMAGE_SIZE = [254, 260, 260]
if ARCHITECTURE == "ct_net" and FILE_TYPE == "feature_tensor":
    IMAGE_SIZE = [130, 512, 13, 13] # This is the shape the model desires. Format: [chunks, channels, f1, f2]
if ARCHITECTURE == "medical_net" and FILE_TYPE == "feature_tensor":
    IMAGE_SIZE = [512, 16, 17, 17] # This is the shape the model desires. Format: [channels, f1, f2, f3]


# Define some architecture specific settings
if ARCHITECTURE == "ct_net":
    CENTER_ON_IMAGENET_MEAN = True
else:
    CENTER_ON_IMAGENET_MEAN = False
if ARCHITECTURE == 'medical_net':
                NUM_CHANNELS = 1
if ARCHITECTURE == 'ct_net':
                NUM_CHANNELS = 3  
            
            
# Check some assumptions
assert STAGE in ["fit", "validate", "test"], "STAGE should be set to either 'fit', 'validate', or 'test'"
assert ARCHITECTURE in ["ct_net", "medical_net"], "ARCHITECTURE should be set to either 'ct_net' or 'medical_net'"
assert CLASSIFIER in ["convs_3d", "attention_pooling", "transformer_attention"], "CLASSIFIER should be set to either '3d_convs' or 'attention_pooling'"
assert len(IMAGE_SIZE) == 3, "We expect IMAGE_SIZE to have 3 dimensions of format [slices, side, side]"
assert IMAGE_SIZE[1] == IMAGE_SIZE[2], "We expect IMAGE_SIZE to have format [slices, side, side] and we only support square images for now"
if ARCHITECTURE == "ct_net":
    assert IMAGE_SIZE[0] % 3 == 0, "The number of slices should be divisible by 3 when using CT-Net." 
assert all(isinstance(size, int) for size in IMAGE_SIZE), "All values in IMAGE_SIZE should be integers"
assert len(POSITIVE_WEIGHTS) == NUM_CLASSES, "The number of positive weights should be equal to the number of classes"
if ARCHITECTURE == 'medical_net':
    assert CLASSIFIER == '3d_convs', "Medical-Net, only works with the 3D convolutions classifier"
if FILE_TYPE == 'feature_tensor':
    assert SAVE_FEATURE_MAPS == False, "When the input is a feature tensor, the feature maps cannot be saved to a .pt file"