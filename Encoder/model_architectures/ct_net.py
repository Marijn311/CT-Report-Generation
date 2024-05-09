import numpy as np
from torchvision import models
import pytorch_lightning as pl
import torch
import torch.nn as nn
from config import *
from utils import save_feature_maps
import einops
from model_architectures.convs_3d_classifier import convs_3d_classifier
from model_architectures.attention_pooling_classifier import attention_pooling_classifier
from model_architectures.transformer_attention_classifier import transformer_attention_classifier


#Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class ct_net_architecture(pl.LightningModule):
    """Class for a PyTorch Lightning CNN-based classification model.
    The CT-Net architecture utilizes a pre-trained ResNet model that has been trained on ImageNet. 
    This images that are fed into this model are pseudo-2D-RGB images.
    This is achieved by dividing the 3D CT images into chunks of 3 slices. The dimension of of the 3 slices is passed as if it were the RGB dimension of a 2D image.
    The input size is [batch_size, nr of chunks of 3 slice, 3, width, height].
    
    Attributes:
        feature_extractor (nn.Sequential): The feature extractor that outputs the states of the last convolutional layer of the ResNet.

    Methods:
        forward(x): Performs the forward pass of the CT-Net architecture.

    References:
        - CT-Net: https://github.com/rachellea/ct-net-models
    """
    
    def __init__(self):
        super(ct_net_architecture, self).__init__()        
        
        # Load the pre-trained model
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT').to("cuda") 
        
        # Define the feature extractor (This outputs the states of the last convolutional layer of the ResNet)
        self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-2])) 

    def forward(self, x):
        """Performs the forward pass of the CT-Net architecture."""
        
        # Merge the batch_size and nr_chunks dimensions
        x = einops.rearrange(x, 'b chunks c y x -> (b chunks) c y x')  
        
        # Pass the input through the feature extractor 
        features = self.feature_extractor(x)         
        
        # Either save the feature maps and return a dummy output, or pass the features to a classifier
        if SAVE_FEATURE_MAPS == True: 
            save_feature_maps(features)  
            logits = torch.zeros((BATCH_SIZE, NUM_CLASSES), requires_grad=True).to("cuda") # Dummy output logits, to make sure that the model wont crash 
        else:
            # Split the batch_size and the nr_chunks into seperate dimensions again. The batch stays the batch dimension, The chunks become the channel dimension in the 3d convs. and the 3d kernel is applied over the c, f1 and f2 dimensions.
            features = einops.rearrange(x, '(b chunks) c f1 f2 -> b chunks c f1 f2', b=BATCH_SIZE) 
            
            # Pass the features to a classifier
            if CLASSIFIER == "convs_3d":
                classifier = convs_3d_classifier()
            if CLASSIFIER == "attention_pooling":
                classifier = attention_pooling_classifier()
            if CLASSIFIER == "transformer_attention":
                classifier = transformer_attention_classifier()
            logits = classifier(features)
        
        return logits
