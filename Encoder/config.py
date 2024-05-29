import torch

# Set seeds
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
STAGE = "fit"                       # Either "fit", "validate", or "test"
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

HOSPITAL_NAMES = ['AMPH', 'AMPH_c1', 'AMPH_c2', 'AMPH_c3', 'AMPH_c4', 'ISAL', 'ISAL_c1', 'ISAL_c2', 'ISAL_c3', 'ISAL_c4', 'LUMC', 'LUMC_c1', 'LUMC_c2', 'LUMC_c3', 'LUMC_c4', 'MAXI', 'MAXI_c1', 'MAXI_c2', 'MAXI_c3', 'MAXI_c4', 'RADB', 'RADB_c1', 'RADB_c2', 'RADB_c3', 'RADB_c4', 'UMG1', 'UMG1_c1', 'UMG1_c2', 'UMG1_c3', 'UMG1_c4', 'UMG2', 'UMG2_c1', 'UMG2_c2', 'UMG2_c3', 'UMG2_c4', 'VUMC', 'VUMC_c1', 'VUMC_c2', 'VUMC_c3', 'VUMC_c4', 'ZUYD', 'ZUYD_c1', 'ZUYD_c2', 'ZUYD_c3', 'ZUYD_c4']
IMAGE_DIR = r"Z:\geselecteerd\AA_Marijn_processed\CT_images_combined_surrogate"
CLASS_NAMES = ["mirroredness", '-90', '-45', '0', '45', '90', "left_upper", "left_lower", "right_upper", "right_middle", "right_lower"]
NUM_CLASSES = len(CLASS_NAMES) 
POSITIVE_WEIGHTS = torch.tensor([1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
DATA_SPLIT = [1810,250,250] 
IMAGE_FILE_NAME = 'ct_net_features_ct_thorax_3_abnormalities.pt'
SAVE_FEATURE_MAPS = False       # Whether to save the feature maps that are produced by the CNN to a .pt file
SAVE_ENCODED_IMAGES = False     # Whether to save the states of the last linear layer in the classifier to a .pt file
BACKGROUND_VALUE = -1000        # -1000 is air in Hounsfield units
DATA_AUGMENT = False 


if IMAGE_FILE_NAME.split('.')[-1] == 'nrrd':
    FILE_TYPE = 'image'                         
elif IMAGE_FILE_NAME.split('.')[-1] == 'pt':
    FILE_TYPE = 'feature_tensor'               
else:
    assert False, "The file type of the given input is not supported. Only .nrrd files are support for images and only .pt files are supported for feature tensors."


# Set image size for the model architectures
if ARCHITECTURE == "ct_net" and FILE_TYPE == "image":
    IMAGE_SIZE = [390, 400, 400]                                        # This is the shape the model desires. Format: [slices, side, side]. The inputted image will be rescaled/cropped/padded during preprocessing to match this size.
if ARCHITECTURE == "medical_net" and FILE_TYPE == "image":
    IMAGE_SIZE = [254, 260, 260]
if ARCHITECTURE == "ct_net" and FILE_TYPE == "feature_tensor":
    IMAGE_SIZE = [130, 512, 13, 13]                                     # This is the shape the model desires. Format: [chunks, channels, f1, f2]
if ARCHITECTURE == "medical_net" and FILE_TYPE == "feature_tensor":
    IMAGE_SIZE = [512, 16, 17, 17]                                      # This is the shape the model desires. Format: [channels, f1, f2, f3]


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
if FILE_TYPE == 'image':
    assert len(IMAGE_SIZE) == 3, "We expect IMAGE_SIZE to have 3 dimensions of format [slices, side, side]"
    assert IMAGE_SIZE[1] == IMAGE_SIZE[2], "We expect IMAGE_SIZE to have format [slices, side, side] and we only support square images for now"
    if ARCHITECTURE == "ct_net":
        assert IMAGE_SIZE[0] % 3 == 0, "The number of slices should be divisible by 3 when using CT-Net."
if FILE_TYPE == 'feature_tensor':
    assert len(IMAGE_SIZE) == 4, "We expect IMAGE_SIZE to have 4 dimensions for feature tensors"
assert all(isinstance(size, int) for size in IMAGE_SIZE), "All values in IMAGE_SIZE should be integers"
assert len(POSITIVE_WEIGHTS) == NUM_CLASSES, "The number of positive weights should be equal to the number of classes"
if ARCHITECTURE == 'medical_net':
    assert CLASSIFIER == '3d_convs', "Medical-Net, only works with the 3D convolutions classifier"
if FILE_TYPE == 'feature_tensor':
    assert SAVE_FEATURE_MAPS == False, "When the input is a feature tensor, the feature maps cannot be saved to a .pt file"
if SAVE_ENCODED_IMAGES == True:
    assert CLASSIFIER == 'convs_3d', "The encoded images can only be saved when using the 3D convolutions classifier, for other classifiers this is not implemented yet"