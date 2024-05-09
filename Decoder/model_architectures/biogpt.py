import pytorch_lightning as pl
from config import *
import einops
from transformers import BioGptForCausalLM, BioGptConfig
from torch import nn

class biogpt_architecture(pl.LightningModule):
    """This class represents a GPT model that has been pretrained on biomedical PubMed data.
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
            
            #fix This model takes token ids as input instead of embedding vectors.
            #the whole benefit of using encoded images in the feature map representation is that they are more similar in shape to embedding vectors.
            #So if the feature map representation is downscaled to NR_IMAGE_TOKENS here, than we loss all this benefit.
            #Hence, using the feature map representation is only possible when the model also accepts embeddings as input.
            #However, I AM NOT SURE IF THAT IS POSSIBLE BECAUSE THE IMAGES WILL BE CONCATENATED WITH THE REPORT TOKENS AND THE REPORT TOKENS ARE EMBEDDED only IN THE MODEL, USING PRETRAINED embedding layers.
            #So i would have to remove the embedding layer from the pretrained model and use this layer to embed the report tokens myself everytime before they go to the model
        
            images = einops.rearrange(images, 'b c f1 f2 -> b c (f1 f2)')       # Flatten the features maps
            images = einops.rearrange(images, 'b c f -> (b f) c')               # Isolate the channel dimension
            images = self.reshape_channels_to_nr_img_tokens(images)             # Use a linear layer to reshape the channel dimension to NR_IMAGE_TOKENS
            images = einops.rearrange(images, '(b f) c -> b c f', b=BATCH_SIZE) # Undo the isolation of the channel dimension
            images = self.reshape_to_hidden_size(images)                        # Reshape the flatted featuremaps to HIDDEN_SIZE
        
        if REPRESENTATION == 'token':
            min_value = 0
            max_value = 40                                                                              # This may change depending on the encoder model and dataset that was used. Set SHOW_DATA_EXAMPLES to True in config.py to see the min and max values of the encoded images. Choose a range that covers most of the values.  
            images = torch.clamp(images, min=min_value, max=max_value)                                  # Clamp the encoded images (which are logits values from a ReLu layer) to a fixed range
            scaled_values = (((images - min_value) / (max_value - min_value)) * (NR_IMG_TOKENS-1))      # Scale the clamped values to the range [0, NR_IMG_TOKENS-1]
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
    