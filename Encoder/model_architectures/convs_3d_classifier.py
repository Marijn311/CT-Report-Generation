import torch.nn as nn
import einops 
from config import *
import pytorch_lightning as pl
from utils import save_encoded_images

#Set seeds
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


class convs_3d_classifier(pl.LightningModule):
    """ A PyTorch Lightning module for the 3D convolutional classifier.

    This classifier consists of 3D convolutional layers followed by fully connected layers.

    Attributes:
        convs_3d (nn.Sequential): Sequential container for the 3D convolutional layers.
        fc_layers (nn.Sequential): Sequential container for the fully connected layers.

    Methods:
        forward(x): Performs the forward pass of the model.

    Notes:
        -  There is the option to save the encoded images (states of the model before the final layer) to a .pt file.
    """
    
    def __init__(self):
        super().__init__()
        nr_input_channels = IMAGE_SIZE[0]
        
        # Define the 3d convolutions
        # The 3D convolutions are performed over the last 3 dimensions of IMAGE_SIZE
        # For CT-Net features this is [c, f1, f2 -> 512, 13, 13]. Because of this large difference in size there is different stride per direction
        if ARCHITECTURE == "ct_net":  self.convs_3d = nn.Sequential(
                nn.Conv3d(in_channels=int(nr_input_channels), out_channels=64, kernel_size = (3,3,3), stride=(3,1,1), padding=0), 
                nn.ReLU(),
                nn.Conv3d(in_channels=64, out_channels=32, kernel_size = (3,3,3), stride=(3,1,1), padding=0), 
                nn.ReLU(),
                nn.Conv3d(in_channels=32, out_channels=16, kernel_size = (3,2,2), stride=(3,2,2), padding=0), 
                nn.ReLU()
                ).to("cuda")
        
        # For Medical-Net the last 3 dimensions are [f1, f2, f3 -> 16, 17, 17]. This is a more regular shape so the same stride is used for all directions. 
        # However, the amount of channels is larger for Medical-Net features so the number of output channels has been adapted.
        if ARCHITECTURE == "medical_net":  self.convs_3d = nn.Sequential(
                nn.Conv3d(in_channels=int(nr_input_channels), out_channels=100, kernel_size = (3,3,3), stride=(2,2,2), padding=(1,0,0)),
                nn.ReLU(),
                nn.Conv3d(in_channels=100, out_channels=50, kernel_size = (3,3,3), stride=(1,1,1), padding=(0,0,0)),
                nn.ReLU(),
                nn.Conv3d(in_channels=50, out_channels=16, kernel_size = (3,3,3), stride=(1,1,1), padding='same'), 
                nn.ReLU()
                ).to("cuda")
      
      
        # Define shape after the 3D convolutions. This shape is needed to intialise the fully connected layers.
        # Unfortunately, the shape after the 3D convolutions is dependent on the exact architecture and image size and needs to be known BEFORE the forward pass.
        # As a solution we manually define the shape here for the cases i am working with but this is not a great general solution.
        if ARCHITECTURE == 'ct_net' and IMAGE_SIZE == [390, 400, 400]:
            convs_output_shape = 4608 #16 *18 *4 *4
        elif ARCHITECTURE == 'medical_net' and IMAGE_SIZE == [254, 260, 260]: 
            convs_output_shape = 3456 #16 *6 *6 *6
        else:
            assert False, "The convs_output_shape for the chosen IMAGE_SIZE and ARCHITECTURE combination has not yet been defined."
            
            
        # Define the fully connected layers
        self.fc_layers = nn.Sequential( 
            nn.Linear(convs_output_shape, 128), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 96), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(96, NUM_CLASSES)).to("cuda")
        
       
   
    def forward(self, x):
        """Perform the forward pass of the model."""

        # Pass the image through the 3D convolutions
        x = self.convs_3d(x) 
        
        # Flatten the output of the 3D convolutions
        x = einops.rearrange(x, 'batch d2 d3 d4 -> b (d2 d3 d4)')
        
        # Pass the flattened output through the fully connected layers. Save the encoded images (states of the model before the final layer) to a .pt file
        if SAVE_ENCODED_IMAGES == True:
            encoded_image_extractor = nn.Sequential(*list(self.fc_layers.children())[:-2]).to("cuda")
            encoded_images = encoded_image_extractor(x)
            save_encoded_images(encoded_images)
            logits = torch.randn(BATCH_SIZE, NUM_CLASSES).to("cuda") # Create dummy tensor so the model wont give errors while saving the encoded images
        else: 
            # Pass the flattened output through the fully connected layers
            logits = self.fc_layers(x)
        
        return logits