from model_architectures.transformer_scratch_utils import Encoder, Decoder
from config import *
import torch
import pytorch_lightning as pl

class transformer_scratch_architecture(pl.LightningModule):
    """
    This class defines the architecture of a transformer-based decoder model that was implemented from scratch.
    This implementation is based on a tutorial by https://www.youtube.com/watch?v=U0s0f995w14
    This model does not load any pretrained weights or embeddings, it is trained from scratch.
   
    Attributes:
        encoder (Encoder): The encoder module of the transformer model.
        decoder (Decoder): The decoder module of the transformer model.

    Methods:
        forward(images, reports): Perform a forward pass through the model.

    """

    def __init__(self):
        super(transformer_scratch_architecture, self).__init__()        
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, images, reports):
        """Perform a forward pass through the model."""
      
        # Remove the last token from the reports. This is done to make sure the reports and generated predictions the same length. Predictions are of size TARGET_LENGTH, which is defined as one token shorter than the reports because the reports contain the sos token at the start, and the predictions do not.	 
        reports = reports[:, :-1]
        
        # Make a mask for the reference reports, to use in the masked multi-headed attention block in the decoder
        report_mask = torch.tril(torch.ones((TARGET_LENGTH, TARGET_LENGTH))).expand(BATCH_SIZE, 1, TARGET_LENGTH, TARGET_LENGTH) # The mask is triangular shape
        
        # Pass the images through the encoder to reshape them to the desired size
        images = self.encoder(images)  
        
        # Pass the reshaped encoded images, the reports, and the report mask through the decoder, to get the logits for the predicted words
        logits = self.decoder(reports, images, report_mask)
        
        return logits
    