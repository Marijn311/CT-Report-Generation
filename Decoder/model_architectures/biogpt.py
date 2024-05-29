import pytorch_lightning as pl
from config import *
import einops
from transformers import BioGptForCausalLM, BioGptConfig
from torch import nn

class biogpt_architecture(pl.LightningModule):
    """
    This class represents a GPT model that has been pretrained on biomedical PubMed data.
    The model uses no cross attention to attend to the encoded images. 
    Instead, the encoded images are fed as the first part of the reference report which the model will try to predict the next word for.

    Attributes:
        model (BioGptForCausalLM): The pretrained BioGptForCausalLM model.
        reshape_channels_to_nr_img_tokens (nn.Linear): Linear layer to reshape the channel dimension of the feature maps to NR_IMG_TOKENS.
        reshape_to_hidden_size (nn.Linear): Linear layer to reshape the flattened feature maps to HIDDEN_SIZE.

    Methods:
        forward(images, reports): Performs a forward pass through the GPT model to get the prediction logits.

    """

    def __init__(self): 
        super().__init__()
        
        # Load the pre-trained bio_gpt model and ajust the vocab size to include the image tokens 
        config = BioGptConfig.from_pretrained("microsoft/biogpt")
        config.vocab_size = VOCAB_SIZE                                                                                  # Needed because the vocab is extended with NR_IMG_TOKENS image tokens
        if REPRESENTATION == 'token':
            config.max_position_embeddings = MAX_LENGTH+IMAGE_SIZE[-1]                                                  # To prevent unnecessary memory/computation usage
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt", config=config, ignore_mismatched_sizes=True) # Due to the extended vocab size the pretrained model will not match the new model exactly. This is why we ignore the mismatched sizes.
       
        # To reshape the encoded images to the correct dimensions 
        if REPRESENTATION == 'feature_map':
            self.reshape_channels_to_nr_img_tokens = nn.Linear(IMAGE_SIZE[1], NR_IMG_TOKENS).to('cuda')    
            self.reshape_to_hidden_size = nn.Linear(IMAGE_SIZE[-1]*IMAGE_SIZE[-2], HIDDEN_SIZE).to('cuda')
        
       
    def forward(self, images, reports):

        # Process the encoded images to the correct dimensions
        if REPRESENTATION == 'feature_map':
            
            #todo implement this
            # The benefit of using the feature map representation is that this shape is more similar to the shape of the embedding vectors.
            # However, the model expects tokens instead of feature maps. 
            # Hence, to make this model compatible with the feature map representation, the embedding layer must be removed from the pretrained model.
            # Preferably, this pretrained embedding layer should be used to then embed the report tokens before they are concatenated with the feature maps representation of the images.
            
            assert False, "Feature map representation is not yet implemented for the BioGPT model. See the todo above"
            images = einops.rearrange(images, 'b c f1 f2 -> b c (f1 f2)')       # Flatten the features maps
            images = einops.rearrange(images, 'b c f -> (b f) c')               # Isolate the channel dimension
            images = self.reshape_channels_to_nr_img_tokens(images)             # Use a linear layer to reshape the channel dimension to NR_IMAGE_TOKENS
            images = einops.rearrange(images, '(b f) c -> b c f', b=BATCH_SIZE) # Undo the isolation of the channel dimension
            images = self.reshape_to_hidden_size(images)                        # Reshape the flatted featuremaps to HIDDEN_SIZE
        
        if REPRESENTATION == 'token':
            images = torch.clamp(images, min=0, max=MAX_ENC_IMG_VALUE)                                  # Clamp the encoded images (which are logits values from a ReLu layer) to a fixed range
            scaled_values = (((images) / (MAX_ENC_IMG_VALUE)) * (NR_IMG_TOKENS-1))                      # Scale the clamped values to the range [0, NR_IMG_TOKENS-1]
            integer_values = torch.round(scaled_values).to(torch.int)                                   # Round the scaled values to integers. The "integer values" are effectively the image tokens.                                 
            image_token_ids = integer_values + 42384                                                    # Add 42434 (original vocab length) to all the integer values to make sure the image tokens are added at the end of the vocab
    

        # Join the image tokens with the report tokens and move the sos token to the start the sequence 
        reports = reports[:, 1:]                                                                                                    # Remove the sos token from the start of the reports
        image_token_ids = torch.cat((torch.ones((BATCH_SIZE, 1), dtype=torch.int64).to('cuda')*SOS_IDX, image_token_ids), dim=1)    # Add the sos token to the image tokens
        input_ids = torch.cat((image_token_ids, reports), dim=1)                                                                    # Join the image inputs with the report tokens
        
        # Make a mask for the reports to prevent attention on padding tokens
        reports_attention_mask = torch.where(input_ids != PAD_IDX, torch.tensor(1), torch.tensor(0))

        # Perform forward pass through the GPT model to get the prediction logits
        logits = self.model(input_ids=input_ids, attention_mask=reports_attention_mask).logits
        
        # Remove the first 96 predictions from the logits. These are the predictions for the image tokens
        logits = logits[:, 96:, :]
        
        #This model generates MAX_LENGTH predictions exluding the sos token. I defined MAX_LENGTH as including the SOS token. Hence I expect the model to generate MAX_LENGTH-1 predictions. Thus I remove the last prediction from the logits.
        logits = logits[:, :-1, :]
        
        return logits 
    