import numpy as np
import torch
from config import *
from monai.networks.nets import resnet18 #pip3 install monai
import logging
import einops
import pytorch_lightning as pl
import torch.nn as nn
from utils import save_feature_maps
from model_architectures.convs_3d_classifier import convs_3d_classifier

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class medical_net_architecture(pl.LightningModule):
    """Class for a PyTorch Lightning CNN-based classification model.
    The MedicalNet architecture utilizes a pre-trained ResNet model that has been trained on multi modality medical image datasets.
    MedicalNet is a ResNet model that uses 3D convolutions to process 3D images.
    The input images have shape [batch_size, 1, nr_slices, height, width]

    Attributes:
        feature_extractor (nn.Sequential): The feature extractor that outputs the states of the last convolutional layer of the ResNet.
    
    Methods:
        forward(x): Performs the forward pass of the Medical-Net architecture.
    
    References:
        - MedicalNet: https://github.com/Project-MONAI/MONAI and https://github.com/Borda/MedicalNet
    
    """
    

    def __init__(self): 
        super().__init__()

                
        #Pretrained weights
        MEDNET_WEIGHTS = r"C:\Users\mborghou\OneDrive - TU Eindhoven\AA_Vakken\AA_Active vakken\Afstuderen\Pretrained_Weights\MedicalNet\resnet_18_23dataset.pth"
        #todo see if i can just use pretrained=true or something instead of this path

        # Load the pre-trained model
        resnet = resnet18(spatial_dims=3, n_input_channels=1, num_classes=NUM_CLASSES).to("cuda")
        net_dict = resnet.state_dict()
        pretrain = torch.load(MEDNET_WEIGHTS)
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
        logging.debug(f"missing in pretrained: {len(missing)}")
        inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
        logging.debug(f"inside pretrained: {len(inside)}")
        unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
        logging.debug(f"unused pretrained: {len(unused)}")
        assert len(inside) > len(missing), f"missing: {missing}, inside: {inside}, unused: {unused}"
        assert len(inside) > len(unused), f"missing: {missing}, inside: {inside}, unused: {unused}"
        pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        resnet.load_state_dict(pretrain['state_dict'], strict=False)
        
        # Define the feature extractor (This outputs the states of the last convolutional layer of the ResNet) 
        self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-2])) 
        
   
    
    def forward(self, x):
        """Performs the forward pass of the Medical-Net architecture."""
        
        # Pass the input through the feature extractor 
        features = self.feature_extractor(x)
        
        # Either save the feature maps and return a dummy output, or pass the features to a classifier
        if SAVE_FEATURE_MAPS == True: 
            save_feature_maps(features)  
            logits = torch.zeros((BATCH_SIZE, NUM_CLASSES), requires_grad=True).to("cuda") # Dummy output logits, to make sure that the model wont crash 
        else:
            if CLASSIFIER == "convs_3d":
                classifier = convs_3d_classifier()
            logits = classifier(features)
        
        return logits
    
    
    
    