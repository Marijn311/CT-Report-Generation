import torch
from transformers import BertTokenizerFast
from tensorflow.keras.preprocessing.text import Tokenizer 
from transformers import AutoTokenizer
import pickle
import os
from transformers import BioGptTokenizer 

# Set the seeds
RANDOM_SEED = 21
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

"""
This is the config file. This file contains nearly all hyperparameters and settings that you might want to change.
Hyperparameters/settings are recognisable by the capital letters.
These Hyperparameters/settings are imported into nearly all other scripts in this project.
These other scripts should never alter the values of the hyperparameters/settings.
By importing the hyperparameters/settings from this file directly, 
there is no need to constantly pass them as arguments to every function or class that needs them.
"""


# Model/Trainer settings
STAGE = 'fit'                           # Either fit, validate, or autoregressive
ARCHITECTURE = 'transformer_scratch'    # Either transformer_scratch, bioclinical_bert, or bio_gpt
BATCH_SIZE = 1
GRAD_ACCUM = 1
NUM_EPOCHS = 250
LEARNING_RATE = 1e-4
MAX_LENGTH = 45 
ACCELERATOR = "gpu"
DEVICES = 1 
PRECISION = 32 
NUM_WORKERS = 0 
MAX_ENC_IMG_VALUE = 25                  # This value is important for correctly attending to the encoded images. Use SHOW_DATA_EXAMPLES to determine a good value. 


# Plotting and printing settings
SHOW_DATA_EXAMPLES = False          # Print the report and image of the sample that is returned by __getitem__ in the data module
SHOW_PREDICTIONS = False            # Shows the available context, the ground truth next word, and the predicted next word during training/validation
PLOT_REPORT_LENGTHS = False         # Plot the distribution of the report lengths in the dataset. Useful for setting the MAX_LENGTH hyperparameter


# Checkpoint paths
CKPT_PATH_TO_LOAD = r"Z:\geselecteerd\ZZ_model_checkpoints\bioclinical_bert\mirroredness 2 a\model-step=3420-val_loss=0.36-val_accuracy=0.89.ckpt"    # Path to the checkpoint to load for validation or autoregressive predictions
CKPT_PATH_TO_SAVE = rf"Z:\geselecteerd\ZZ_model_checkpoints\{ARCHITECTURE}"                                                             # Path to the directory where the checkpoints will be saved during training
AUTOREGRESSIVE_DATASET = 'val'                                                                                                          # Dataset to use for autoregressive predictions, either train, val, test.


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
|   |   reports.docs
│   [hospital_name_2]
|   |   [hospital_name_2]_[pseudo_id_1]
|   |   │   [IMAGE_FILE_NAME]
|   |   [hospital_name_2]_[pseudo_id_2]
|   |   │   [IMAGE_FILE_NAME]
|   |   ...
|   |   labels.xlsx
|   |   reports.docs
|   ...
|   all_labels.xlsx
|   all_reports.docs
|   SARLE_dataset.xlsx

"""                                                       

HOSPITAL_NAMES = ['AMPH', 'AMPH_c1', 'AMPH_c2', 'AMPH_c3', 'AMPH_c4', 'ISAL', 'ISAL_c1', 'ISAL_c2', 'ISAL_c3', 'ISAL_c4', 'LUMC', 'LUMC_c1', 'LUMC_c2', 'LUMC_c3', 'LUMC_c4', 'MAXI', 'MAXI_c1', 'MAXI_c2', 'MAXI_c3', 'MAXI_c4', 'RADB', 'RADB_c1', 'RADB_c2', 'RADB_c3', 'RADB_c4', 'UMG1', 'UMG1_c1', 'UMG1_c2', 'UMG1_c3', 'UMG1_c4', 'UMG2', 'UMG2_c1', 'UMG2_c2', 'UMG2_c3', 'UMG2_c4', 'VUMC', 'VUMC_c1', 'VUMC_c2', 'VUMC_c3', 'VUMC_c4', 'ZUYD', 'ZUYD_c1', 'ZUYD_c2', 'ZUYD_c3', 'ZUYD_c4']
IMAGE_DIR = r"Z:\geselecteerd\AA_Marijn_processed\CT_images_combined_surrogate"
DATA_SPLIT = [1810,250,250]
IMAGE_FILE_NAME = 'ct_net_3d_convs_combined_1_a.pt'
REPRESENTATION = 'token'                                                                    # Either token or feature_map
TASK = 'combined_surrogate'                                                                 # For naming the vocab dict file

if REPRESENTATION == 'feature_map':
    IMAGE_SIZE = [BATCH_SIZE, 130, 512, 13, 13]                                             # Size of the tensor in IMAGE_FILE_NAME
if REPRESENTATION == 'token':
    IMAGE_SIZE = [BATCH_SIZE, 96]                                                           # Size of the tensor in IMAGE_FILE_NAME


# Parameters for the transformer_scratch model 
if ARCHITECTURE == 'transformer_scratch':                                                   # 4.1M parameters with the settings below
    
    # Architecture parameters
    HIDDEN_SIZE = 256
    NUM_LAYERS = 6 
    FORWARD_EXPANSION = 4                                                                   # Factor with which HIDDEN_SIZE is multiplied in the feedforward layer (to add more trainable parameters / computational power to the model)
    NR_HEADS = 8 
    DROPOUT = 0.1
    NR_IMG_TOKENS = 100                                                                     # Number of tokens that are reserved for the images. This range of the image tokens, NOT the amount of image tokens. The amount of image tokens is IMAGE_SIZE[-1]
    TARGET_LENGTH = torch.tensor(MAX_LENGTH-1)                                              # The target length is the maximum length of the reports minus 1, because the reports contain the sos token at the start, and the predictions do not.
    
    # Tokenizer parameters
    TOKENIZER = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*/:;<=>?@[\\]^_`{|}~\t\n')    # Filters are set to remove all punctuation DO NOT put [.] or [,] or [+] or [-] in there
    PAD_IDX = 0
    SOS_IDX = 8
    EOS_IDX = 9 
    VOCAB_SIZE = 190
    assert os.path.exists(f'vocab_dict_scratch_{TASK}.pkl'), "Vocab dict file does not exist yet. Outcomment this assert statement and the 2 lines below, set mode to fit and all the plotting/printing to False. Run main.py again, during initialisation the vocab file will be saved. Then you will encounter an error again because the model was not initialized with vocab dict. So you have to run main.py again to load the model with the vocab dict file this time."
    with open(f'vocab_dict_scratch_{TASK}.pkl', 'rb') as f:
            VOCAB_DICT = pickle.load(f)

    
# Parameters for the bioclinical_bert model
if ARCHITECTURE == 'bioclinical_bert':                                                          #159M parameters with the settings below
    
    # Architecture parameters
    HIDDEN_SIZE = 768                                                                           # Needed to embed the image into the same size as the bert embeddings
    NR_IMG_TOKENS = 100                                                                         # Number of tokens that are reserved for the images. This range of the image tokens, NOT the amount of image tokens. The amount of image tokens is IMAGE_SIZE[-1]
    
    # Tokenizer parameters
    TOKENIZER = BertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    PAD_IDX = 0
    SOS_IDX = 101
    EOS_IDX = 102 
    VOCAB_DICT = TOKENIZER.vocab 
    VOCAB_SIZE = len(VOCAB_DICT)


# Parameters for the bio_gpt model
if ARCHITECTURE == 'bio_gpt':                                                                   #346M parameters with the settings below
   
    # Architecture parameters
    HIDDEN_SIZE = 1024                                                                          # Needed to embed the image into the same size as the gpt embeddings
    
    # Tokenizer parameters
    TOKENIZER = BioGptTokenizer.from_pretrained("microsoft/biogpt")                     
    PAD_IDX = 1
    SOS_IDX = 2 
    EOS_IDX = 1                                                                                 # This tokenizer does not have an eos token so we use the (first occurance of) pad token as eos
    NR_IMG_TOKENS = 100
    for i in range(NR_IMG_TOKENS):                                                              # Add x image tokens to the tokenizer
        TOKEN   = f"<img_{i}>"
        TOKENIZER.add_tokens(TOKEN)
    VOCAB_DICT = TOKENIZER.get_vocab() 
    VOCAB_SIZE = len(VOCAB_DICT) 
    

# Check some assumptions
assert STAGE in ["fit", "validate", "autoregressive"], "STAGE should be set to either 'fit', 'validate', or 'autoregressive'"
assert ARCHITECTURE in ["transformer_scratch", "bioclinical_bert", "bio_gpt"], f"ARCHITECTURE {ARCHITECTURE} is not supported"
assert REPRESENTATION in ["token", "feature_map"], f"REPRESENTATION {REPRESENTATION} is not supported"
if ARCHITECTURE == 'transformer_scratch':
    assert HIDDEN_SIZE % NR_HEADS == 0, "HIDDEN_SIZE needs to be divisible by NR_HEADS"
